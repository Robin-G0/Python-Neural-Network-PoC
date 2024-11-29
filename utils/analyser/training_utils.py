import numpy as np

def print_progress_bar(current, total, length=40, prefix='', suffix='', fill='█', print_end="\r"):
    """
    Prints an ASCII progress bar.
    Args:
        current: Current progress (e.g., batch/epoch index).
        total: Total steps (e.g., total batches/epochs).
        length: Length of the progress bar.
        prefix: Text prefix.
        suffix: Text suffix.
        fill: Character to use for the filled portion.
        print_end: End character (default is carriage return for updating).
    """
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if current == total:
        print()

def adjust_learning_rate(learning_rate, epoch, decay_rate=0.5, decay_step=40):
    """
    Adjusts the learning rate based on the epoch number.

    Args:
        learning_rate: Initial learning rate.
        epoch: Current epoch number.
        decay_rate: Rate of decay.
        decay_step: Step size for decay.

    Returns:
        Adjusted learning rate.
    """
    return learning_rate * (decay_rate ** (epoch // decay_step))

def train_network_multithreaded(network, data, learning_rate=0.005, epochs=10, batch_size=72, updates_queue=None, stop_flag=None):
    """
    Trains the network with a progress bar and multithreading updates.

    Args:
        network: Neural network structure.
        data: Training data in the form of (inputs, targets) tuples.
        learning_rate: Learning rate for gradient descent.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        updates_queue: Queue for sending updates to the main thread.
        stop_flag: Event flag to stop training early.

    Returns:
        None
    """
    num_samples = len(data)
    total_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        current_lr = adjust_learning_rate(learning_rate, epoch)
        print(f"Epoch {epoch + 1}/{epochs} - Learning Rate: {current_lr:.6f}")
        if stop_flag and stop_flag.is_set():
            break

        np.random.shuffle(data)
        epoch_loss, correct_predictions = 0, 0

        for i, start_idx in enumerate(range(0, num_samples, batch_size)):
            if stop_flag and stop_flag.is_set():
                break

            batch = data[start_idx:start_idx + batch_size]
            batch_loss = 0

            for inputs, targets in batch:
                outputs = forward_pass(network, inputs)
                loss = compute_loss(outputs, targets)
                gradients = compute_gradients(network, outputs, targets, regularization=0.001)

                # Update weights with scaled learning rate
                update_weights(network, gradients, current_lr)

                batch_loss += loss
                if np.argmax(outputs) == np.argmax(targets):
                    correct_predictions += 1

            epoch_loss += batch_loss / len(batch)
            print_progress_bar(i + 1, total_batches, prefix='Progress:', suffix=f'Loss: {batch_loss / len(batch):.4f} / Accuracy: {correct_predictions / num_samples:.4f}')

            if updates_queue:
                updates_queue.put((epoch_loss / (i + 1), correct_predictions / num_samples))

        epoch_loss /= num_samples
        epoch_accuracy = correct_predictions / num_samples
        if updates_queue:
            updates_queue.put((epoch_loss, epoch_accuracy))

def forward_pass(network, inputs):
    """
    Simulates a forward pass through the network.

    Args:
        network: Neural network structure.
        inputs: Input data.

    Returns:
        Outputs of the network.
    """
    layer_input = inputs
    # print(f"Forward Pass - Initial Input: {layer_input[:10]}... (truncated), Shape: {layer_input.shape}") # Debug
    
    for idx, layer in enumerate(network['layers']):
        weights = layer['weights']
        biases = layer['biases']
        activation = layer['activation']

        # Calculate raw pre-activation outputs
        z = np.dot(weights, layer_input) + biases
        # print(f"Layer {idx + 1}: z (pre-activation) min: {np.min(z)}, max: {np.max(z)}, shape: {z.shape}") # Debug

        # Store the raw outputs ('z') and the input to this layer
        layer['z'] = z
        layer['input'] = layer_input

        # Apply activation function
        layer_input = apply_activation(z, activation)
        # print(f"Layer {idx + 1}: output (post-activation) min: {np.min(layer_input)}, max: {np.max(layer_input)}, shape: {layer_input.shape}") # Debug

    # print(f"Forward Pass - Final Output: {layer_input[:10]}... (truncated), Shape: {layer_input.shape}") # Debug
    return layer_input

def apply_activation(z, activation):
    """
    Applies the specified activation function.

    Args:
        z: Input to the activation function.
        activation: Type of activation function.

    Returns:
        Activated output.
    """
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'softmax':
        z -= np.max(z)  # Subtract max for numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def compute_loss(outputs, targets):
    """
    Computes the cross-entropy loss.

    Args:
        outputs: Predicted outputs of the network.
        targets: Target labels.

    Returns:
        Cross-entropy loss.
    """
    epsilon = 1e-7  # Prevent log(0)
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    if isinstance(targets, int):
        # Convert to one-hot encoding if targets are integers
        one_hot_targets = np.eye(outputs.shape[0])[targets]
        return -np.sum(one_hot_targets * np.log(outputs)) / outputs.shape[0]
    return -np.sum(targets * np.log(outputs)) / targets.shape[0]

def compute_gradients(network, outputs, targets, regularization):
    """
    Computes the gradients for each layer in the network.
    
    Args:
        network: Neural network structure.
        outputs: Outputs of the network.
        targets: Target labels.
        regularization: L2 regularization coefficient.
    
    Returns:
        Gradients for each layer.
    """
    gradients = []
    error = outputs - targets

    for i, layer in reversed(list(enumerate(network['layers']))):
        if not isinstance(layer, dict):
            raise TypeError(f"Expected layer to be a dict, but got {type(layer)}. Layer: {layer}")
        
        activation = layer['activation']
        weights = layer['weights']

        # Compute gradient of activation
        grad_activation = np.where(layer['z'] > 0, 1, 0) if activation == 'relu' else 1
        delta = error * grad_activation
        grad_weights = np.outer(delta, layer['input']) + 2 * regularization * weights
        grad_biases = delta

        # Normalize gradients
        grad_norm = np.linalg.norm(grad_weights)
        if grad_norm > 1.0:
            grad_weights /= grad_norm
        grad_biases = np.clip(grad_biases, -1.0, 1.0)
        grad_weights = np.clip(grad_weights, -1.0, 1.0)

        gradients.insert(0, {'weights': grad_weights, 'biases': grad_biases})
        error = np.dot(weights.T, delta)

    return gradients

def update_weights(network, gradients, learning_rate, clip_value=1.0):
    """
    Updates the weights of the network with gradient clipping.

    Args:
        network: Neural network structure.
        gradients: Gradients for each layer.
        learning_rate: Learning rate for gradient descent.
        clip_value: Maximum allowed value for gradients.

    Returns:
        None
    """
    for i, (layer, grad) in enumerate(zip(network['layers'], gradients)):
        grad_weights = np.clip(grad['weights'], -clip_value, clip_value)
        grad_biases = np.clip(grad['biases'], -clip_value, clip_value)
        layer['weights'] -= learning_rate * grad_weights
        layer['biases'] -= learning_rate * grad_biases
        # print(f"Layer {i + 1}: Updated Weights min: {layer['weights'].min()}, max: {layer['weights'].max()}")  # Debug

def regularization_term(network, regularization):
    """
    Computes the L2 regularization term.

    Args:
        network: Neural network structure.
        regularization: L2 regularization coefficient.

    Returns:
        Regularization term.
    """
    reg_term = sum(np.sum(layer['weights'] ** 2) for layer in network['layers'])
    reg_term *= regularization
    print(f"Regularization Term: {reg_term:.4f}")  # Debug
    return reg_term

def one_hot_encode(labels, num_classes):
    """
    Converts integer labels into one-hot encoded vectors.
    Args:
        labels: List or array of integer labels.
        num_classes: Number of classes.
    Returns:
        One-hot encoded NumPy array.
    """
    return np.eye(num_classes)[labels]
