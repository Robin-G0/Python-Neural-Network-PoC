import numpy as np

def interpret_decision(result_path):
    """
    Interprets the sequence of predictions from the neural networks.

    Args:
        result_path (list): List of binary decisions in the order of neural networks.

    Returns:
        str: Final interpreted result.
    """
    if len(result_path) == 0 or result_path[0] == 0:
        return "Nothing"
    if len(result_path) > 1 and result_path[1] == 0:
        return "Stalemate"
    if len(result_path) > 2 and result_path[2] == 0:
        if len(result_path) > 3 and result_path[3] == 0:
            return "Check White"
        elif len(result_path) > 3 and result_path[3] == 1:
            return "Check Black"
        return "Check"
    if len(result_path) > 3 and result_path[3] == 0:
        return "Checkmate White"
    return "Checkmate Black"

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
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
