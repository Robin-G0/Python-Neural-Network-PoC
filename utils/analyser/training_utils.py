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

def train_network_multithreaded(network, data, learning_rate=0.001, epochs=10, batch_size=72, updates_queue=None, stop_flag=None):
    """
    Trains the network with a progress bar and multithreading updates.
    """
    print(f"Training data sample min: {np.min([inputs.min() for inputs, _ in data])}, max: {np.max([inputs.max() for inputs, _ in data])}")  # Debug
    num_samples = len(data)
    total_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            print("Training stopped.")
            break

        np.random.shuffle(data)  # Shuffle data for better generalization
        epoch_loss = 0
        correct_predictions = 0

        print(f"Epoch {epoch + 1}/{epochs}:")
        for i, start_idx in enumerate(range(0, num_samples, batch_size)):
            if stop_flag and stop_flag.is_set():
                print("\nTraining stopped during batch.")
                break

            batch = data[start_idx:start_idx + batch_size]
            batch_loss = 0

            for inputs, targets in batch:
                # Forward pass
                outputs = forward_pass(network, inputs)

                # Compute loss and add regularization term
                loss = compute_loss(outputs, targets) + regularization_term(network, 0.001)
                batch_loss += loss

                # Backward pass
                gradients = compute_gradients(network, outputs, targets, 0.001)

                # Update weights
                update_weights(network, gradients, learning_rate)

                # Track accuracy
                if np.argmax(outputs) == targets:
                    correct_predictions += 1

            epoch_loss += batch_loss / len(batch)

            # Update progress bar
            print_progress_bar(i + 1, total_batches, prefix="Batch Progress", suffix="Complete")

            # Send batch update to the queue
            if updates_queue:
                batch_accuracy = correct_predictions / num_samples
                updates_queue.put((epoch_loss / (i + 1), batch_accuracy))

        epoch_loss /= num_samples
        epoch_accuracy = correct_predictions / num_samples
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Send epoch update to the queue
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
    print(f"Forward Pass - Initial Input: {layer_input[:10]}... (truncated), Shape: {layer_input.shape}")
    
    for idx, layer in enumerate(network['layers']):
        weights = layer['weights']
        biases = layer['biases']
        activation = layer['activation']

        # Calculate raw pre-activation outputs
        z = np.dot(weights, layer_input) + biases
        print(f"Layer {idx + 1}: z (pre-activation) min: {np.min(z)}, max: {np.max(z)}, shape: {z.shape}")

        # Store the raw outputs ('z') and the input to this layer
        layer['z'] = z
        layer['input'] = layer_input

        # Apply activation function
        layer_input = apply_activation(z, activation)
        print(f"Layer {idx + 1}: output (post-activation) min: {np.min(layer_input)}, max: {np.max(layer_input)}, shape: {layer_input.shape}")

    print(f"Forward Pass - Final Output: {layer_input[:10]}... (truncated), Shape: {layer_input.shape}")
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
    Computes the loss between the network's outputs and the targets.

    Args:
        outputs: Outputs from the network.
        targets: Expected targets.

    Returns:
        Loss value.
    """
    loss = np.mean((outputs - targets) ** 2)
    print(f"Compute Loss - Loss: {loss:.4f}")  # Debug
    return loss

def compute_gradients(network, outputs, targets, regularization):
    """
    Computes gradients for backpropagation.

    Args:
        network: Neural network structure.
        outputs: Outputs from the network.
        targets: Expected targets.
        regularization: L2 regularization coefficient.

    Returns:
        Gradients for each layer.
    """
    gradients = []
    error = outputs - targets
    print(f"Gradients - Initial Error min: {error.min()}, max: {error.max()}")  # Debug

    for i, layer in reversed(list(enumerate(network['layers']))):
        activation = layer['activation']
        weights = layer['weights']

        # Compute gradient of activation function
        if activation == 'relu':
            grad_activation = np.where(layer['z'] > 0, 1, 0)
        elif activation == 'softmax':
            grad_activation = np.ones_like(layer['z'])
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        delta = error * grad_activation
        grad_weights = np.outer(delta, layer['input']) + 2 * regularization * weights
        grad_biases = delta

        # Clip gradients to prevent exploding gradients
        grad_weights = np.clip(grad_weights, -1.0, 1.0)
        grad_biases = np.clip(grad_biases, -1.0, 1.0)

        print(f"Layer {i + 1}: Grad Weights min: {grad_weights.min()}, max: {grad_weights.max()}")  # Debug
        print(f"Layer {i + 1}: Grad Biases min: {grad_biases.min()}, max: {grad_biases.max()}")  # Debug

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
    """
    for i, (layer, grad) in enumerate(zip(network['layers'], gradients)):
        grad_weights = np.clip(grad['weights'], -clip_value, clip_value)
        grad_biases = np.clip(grad['biases'], -clip_value, clip_value)
        layer['weights'] -= learning_rate * grad_weights
        layer['biases'] -= learning_rate * grad_biases
        print(f"Layer {i + 1}: Updated Weights min: {layer['weights'].min()}, max: {layer['weights'].max()}")  # Debug

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
