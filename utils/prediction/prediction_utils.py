import numpy as np

def interpret_decision(result_path):
    """
    Interprets the sequence of predictions from the neural networks.

    Args:
        result_path (list): List of decisions in prediction order:
            [Something/Nothing, Check/Checkmate/Stalemate, White/Black]

    Returns:
        str: Final interpreted result.
    """
    if not result_path or len(result_path) < 1:
        return "Error"

    state_decision = result_path[0]

    if state_decision == 3:  # Nothing
        return "Nothing"
    elif state_decision == 2:  # Stalemate
        return "Stalemate"

    if len(result_path) < 2:
        return "Error: Missing Color Decision"

    color_decision = result_path[1]

    if state_decision == 0:  # Check
        if color_decision == 0:
            return "Check Black"
        elif color_decision == 1:
            return "Check White"

    elif state_decision == 1:  # Checkmate
        if color_decision == 0:
            return "Checkmate Black"
        elif color_decision == 1:
            return "Checkmate White"

    return "Error: Invalid Decision"

def predict(network, inputs):
    """
    Generates a prediction for the given inputs using the provided network.

    Args:
        network (dict): The neural network structure.
        inputs (np.ndarray): The input feature vector.

    Returns:
        np.ndarray: The output probabilities or values from the network.
    """
    layer_input = inputs
    for layer in network['layers']:
        weights = layer['weights']
        biases = layer['biases']
        activation = layer['activation']
        z = np.dot(weights, layer_input) + biases
        layer_input = apply_activation(z, activation)
    return layer_input

def apply_activation(z, activation):
    """
    Applies the specified activation function.

    Args:
        z (np.ndarray): The pre-activation outputs.
        activation (str): The activation function type.

    Returns:
        np.ndarray: Activated output.
    """
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'softmax':
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
