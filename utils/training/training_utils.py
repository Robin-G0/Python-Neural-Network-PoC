import numpy as np
import sys

def print_progress_bar(current, total, length=40, prefix='', suffix='', fill='█', print_end="\r"):
    """
    Print a progress bar to the console.
    """
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end, file=sys.stderr)
    if current == total:
        print("", file=sys.stderr)

def adjust_learning_rate(learning_rate, epoch, decay_rate=0.5, decay_step=40):
    """
    Adjusts the learning rate based on the current epoch.
    """
    return learning_rate * (decay_rate ** (epoch // decay_step))

def train_network_multithreaded(network, data, learning_rate=0.005, epochs=10, batch_size=500, updates_queue=None, stop_flag=None):
    """
    Trains a neural network using mini-batch gradient descent.

    Args:
        network (dict): Dictionary containing the network configuration.
        data (list): List of tuples containing inputs and labels.
        learning_rate (float): Learning rate for the update.
        epochs (int): Number of epochs to train the network.
        batch_size (int): Size of the mini-batches.
        updates_queue (queue.Queue): Queue to store the training updates.

    Returns:
        float: Loss of the network after training.
        float: Accuracy of the network after training.
    """
    num_samples = len(data)
    total_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            print("Training stopped by user.", file=sys.stderr)
            return
        current_lr = adjust_learning_rate(learning_rate, epoch)
        print(f"Epoch {epoch + 1}/{epochs} - Learning Rate: {current_lr:.6f}", file=sys.stderr)

        np.random.shuffle(data)
        epoch_loss, correct_predictions = 0.0, 0

        for i, start_idx in enumerate(range(0, num_samples, batch_size)):
            if stop_flag and stop_flag.is_set():
                print("Training stopped by user.", file=sys.stderr)
                return
            batch = data[start_idx:start_idx + batch_size]
            batch_loss = 0.0

            for inputs, targets in batch:
                outputs = forward_pass(network, inputs)
                loss = compute_loss(outputs, targets)
                gradients = compute_gradients(network, outputs, targets, regularization=0.001)
                update_weights(network, gradients, current_lr)

                batch_loss += loss
                correct_predictions += int((outputs.round() == targets).all())

            epoch_loss += batch_loss / len(batch)
            if stop_flag and stop_flag.is_set():
                print("Training stopped by user.", file=sys.stderr)
                return
            print_progress_bar(
                i + 1,
                total_batches,
                prefix='Progress:',
                suffix=f'Loss: {float(batch_loss / len(batch)):.4f} / Accuracy: {correct_predictions / num_samples:.4f}'
            )

            if updates_queue:
                updates_queue.put((epoch_loss / (i + 1), correct_predictions / num_samples))

        epoch_loss /= num_samples
        epoch_accuracy = correct_predictions / num_samples
        if updates_queue:
            updates_queue.put((epoch_loss, epoch_accuracy))

def forward_pass(network, inputs):
    """
    Performs a forward pass through the network.

    Args:
        network (dict): Dictionary containing the network configuration.
        inputs (np.ndarray): Input to the network.

    Returns:
        np.ndarray: Output of the network.
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

    Args:
        z (np.ndarray): Input to the activation function.
        activation (str): Name of the activation function.

    Returns:
        np.ndarray: Output of the activation function.
    """
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def compute_loss(outputs, targets):
    """
    Computes binary cross-entropy loss for scalar targets.

    Args:
        outputs (np.ndarray): Output of the network.
        targets (np.ndarray): Target values for the network.

    Returns:
        float: Binary cross-entropy loss.
    """
    epsilon = 1e-7  # avoid log(0)
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))

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
        elif activation == 'softmax':
            softmax = apply_activation(layer['z'], 'softmax')
            grad_activation = softmax * (1 - softmax)
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

    Returns:
        dict: Dictionary mapping labels to their corresponding indices.
    """
    return {
        "something_vs_nothing": {"Something": 1, "Nothing": 0},
        "check_vs_stalemate": {"Check": 1, "Stalemate": 0},
        "checkmate_vs_check": {"Checkmate": 1, "Check": 0},
        "white_vs_black": {"White": 1, "Black": 0},
        "check_checkmate_stalemate": {"Check": 0, "Checkmate": 1, "Stalemate": 2}
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
