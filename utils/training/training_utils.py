import numpy as np
import sys
import copy
from utils.training.progress import print_progress_bar
from utils.training.learning_rate import adjust_learning_rate

def apply_dropout(inputs, dropout_rate):
    """
    Applies dropout to the input layer during training.
    Args:
        inputs (np.ndarray): Input array.
        dropout_rate (float): Probability of dropping a unit.
    Returns:
        np.ndarray: The input with dropout applied.
    """
    if dropout_rate <= 0.0 or dropout_rate >= 1.0:
        return inputs  # No dropout if rate is invalid
    dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=inputs.shape)
    return inputs * dropout_mask

def train_network_multithreaded(network, data, initial_learning_rate=0.001, epochs=20, batch_size=500, 
                                updates_queue=None, stop_flag=None, lr_strategy="reduce_on_plateau", lr_params=None, dropout_rate=0.2):
    """
    Entraîne un réseau de neurones avec dropout et mise à jour du taux d'apprentissage.

    Args:
        network (dict): Réseau à entraîner.
        data (list): Données au format (inputs, labels).
        initial_learning_rate (float): Taux d'apprentissage initial.
        epochs (int): Nombre d'époques.
        batch_size (int): Taille des mini-lots.
        updates_queue (queue.Queue): File d'attente pour transmettre les mises à jour.
        stop_flag (threading.Event): Indicateur d'arrêt.
        lr_strategy (str): Stratégie de réglage du taux d'apprentissage.
        lr_params (dict): Paramètres supplémentaires pour l'ajustement du taux d'apprentissage.
        dropout_rate (float): Taux de dropout appliqué aux couches cachées.
    """
    if lr_params is None:
        lr_params = {}

    num_samples = len(data)
    total_batches = (num_samples + batch_size - 1) // batch_size

    # Split data into train/validation
    split_idx = int(num_samples * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    learning_rate = initial_learning_rate
    best_val_acc = 0
    last_plateau_epoch = -1
    best_epoch_loss = float('inf')
    best_epoch_loss_val = 1.0
    best_epoch = -1
    best_network_state = None
    consecutive_increase_count = 0

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            print("Training stopped by user.", file=sys.stderr)
            break

        # Adjust Learning Rate
        if lr_strategy == "reduce_on_plateau":
            learning_rate, last_plateau_epoch = adjust_learning_rate(
                learning_rate, epoch, strategy=lr_strategy,
                val_acc=best_val_acc, last_plateau_epoch=last_plateau_epoch, **lr_params
            )
        else:
            learning_rate = adjust_learning_rate(
                learning_rate, epoch, strategy=lr_strategy, **lr_params
            )

        print(f"Epoch {epoch + 1}/{epochs} - Learning Rate: {learning_rate:.6f}", file=sys.stderr)
        np.random.shuffle(train_data)

        epoch_loss, correct_predictions = 0.0, 0

        for start_idx in range(0, len(train_data), batch_size):
            if stop_flag and stop_flag.is_set():
                print("Training stopped by user.", file=sys.stderr)
                break

            batch = list(train_data)[start_idx:start_idx + batch_size]
            batch_loss = 0.0

            for inputs, targets in batch:
                # Apply dropout only during training
                inputs_dropped = apply_dropout(inputs, dropout_rate)
                outputs = forward_pass(network, inputs_dropped)
                loss = compute_loss(outputs, targets)
                gradients = compute_gradients(network, outputs, targets, regularization=0.001)
                update_weights(network, gradients, learning_rate)

                batch_loss += loss
                correct_predictions += int((outputs.round() == targets).all())

            print_progress_bar(
                start_idx // batch_size + 1, total_batches, prefix='Progress:', suffix='Complete', length=20
            )
            epoch_loss += batch_loss / len(batch)

        # Evaluate validation accuracy
        val_outputs = np.array([forward_pass(network, inputs) for inputs, _ in val_data])
        val_targets = np.array([targets for _, targets in val_data])

        if val_outputs.ndim > 1 and val_outputs.shape[1] > 1:  # Multi-class classification
            val_predictions = np.argmax(val_outputs, axis=1)
            val_target_labels = np.argmax(val_targets, axis=1)
        else:  # Binary classification
            val_predictions = (val_outputs >= 0.5).astype(int).ravel()
            val_target_labels = val_targets.astype(int).ravel()

        val_correct = np.sum(val_predictions == val_target_labels)
        val_accuracy = val_correct / len(val_data)
        best_val_acc = max(best_val_acc, val_accuracy)

        if updates_queue:
            updates_queue.put((epoch_loss / total_batches, correct_predictions / len(train_data), val_accuracy))

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / total_batches:.4f} - "
              f"Train Accuracy: {correct_predictions / len(train_data):.4f} - Val Accuracy: {val_accuracy:.4f}",
              file=sys.stderr)

        # Check for early stopping
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch = epoch
            best_network_state = copy.deepcopy(network)
            best_epoch_loss_val = epoch_loss / total_batches
            consecutive_increase_count = 0  # Reset the counter
        elif epoch_loss > best_epoch_loss:
            consecutive_increase_count += 1
            if consecutive_increase_count >= 3:
                print(f"Stopping early at epoch {epoch + 1} due to 3 consecutive increases in loss.", file=sys.stderr)
                break
        else:
            consecutive_increase_count = 0  # Reset the counter if loss does not increase

    # Restore the best network state
    if best_network_state:
        network.update(best_network_state)
        print(f"Restored network state from epoch {best_epoch + 1} with lowest loss {best_epoch_loss_val:.4f}.", file=sys.stderr)

def forward_pass(network, inputs):
    """
    Performs a forward pass through the network.
    """
    layer_input = inputs
    for layer in network['layers']:
        weights = layer['weights']
        biases = layer['biases']
        activation = layer['activation']
        z = np.dot(weights, layer_input) + biases
        layer['z'] = z
        layer['input'] = layer_input
        layer_input = apply_activation(z, activation)
    return layer_input

def apply_activation(z, activation):
    """
    Applies the specified activation function to the input.
    """
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'softmax':
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def compute_loss(outputs, targets):
    """
    Computes the categorical cross-entropy loss for multi-class classification
    or binary cross-entropy for binary classification.
    """
    epsilon = 1e-7  # avoid log(0)
    outputs = np.clip(outputs, epsilon, 1 - epsilon)

    if len(outputs.shape) == 1 or outputs.shape[1] == 1:  # Binary classification
        return -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
    else:  # Multi-class classification
        return -np.mean(np.sum(targets * np.log(outputs), axis=1))

def compute_gradients(network, outputs, targets, regularization):
    """
    Computes gradients for each layer in the network.

    Args:
        network (dict): Dictionary containing the network configuration.
        outputs (np.ndarray): Output of the network.
        targets (np.ndarray): Target values for the network.
        regularization (float): Regularization parameter.

    Returns:
        list: List of dictionaries containing gradients for each layer.
    """
    gradients = []
    error = outputs - targets

    for layer in reversed(network['layers']):
        activation = layer['activation']
        weights = layer['weights']

        if activation == 'relu':
            grad_activation = np.where(layer['z'] > 0, 1, 0)
        elif activation == 'sigmoid':
            sigmoid = apply_activation(layer['z'], 'sigmoid')
            grad_activation = sigmoid * (1 - sigmoid)
        else:
            grad_activation = 1

        delta = error * grad_activation
        grad_weights = np.outer(delta, layer['input']) + 2 * regularization * weights
        grad_biases = delta

        gradients.insert(0, {'weights': grad_weights, 'biases': grad_biases})
        error = np.dot(weights.T, delta)

    return gradients

def update_weights(network, gradients, learning_rate, clip_value=2.0):
    """
    Updates the weights and biases of the network using the computed gradients.

    Args:
        network (dict): Dictionary containing the network configuration.
        gradients (list): List of dictionaries containing gradients for each layer.
        learning_rate (float): Learning rate for the update.
        clip_value (float): Value to clip the gradients.
    """
    for layer, grad in zip(network['layers'], gradients):
        if not isinstance(grad, dict):
            raise TypeError(f"Expected gradient to be a dictionary, but got {type(grad)}: {grad}")

        grad_weights = np.clip(grad['weights'], -clip_value, clip_value)
        grad_biases = np.clip(grad['biases'], -clip_value, clip_value)
        layer['weights'] -= learning_rate * grad_weights
        layer['biases'] -= learning_rate * grad_biases

def get_label_map():
    """
    Returns a dictionary mapping labels to their corresponding indices.
    """
    return {
        "check_checkmate_stalemate_nothing": {"Check": 0, "Checkmate": 1, "Stalemate": 2, "Nothing": 3},
        "white_vs_black": {"White": 1, "Black": 0}
    }

def filter_data_by_labels(data, label_map):
    """
    Filters the data based on the valid labels for the network.
    
    Args:
        data (list): List of tuples containing inputs and labels.
        label_map (dict): Dictionary mapping labels to their corresponding indices.
    
    Returns:
        list: List of tuples containing inputs and labels.
    """
    valid_labels = set(label_map.keys())
    return [(inputs, label) for inputs, label in data if label in valid_labels]

def encode_one_hot(labels, num_classes):
    """
    Converts integer labels into one-hot encoded format.
    """
    one_hot = np.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        one_hot[idx, label] = 1
    return one_hot
