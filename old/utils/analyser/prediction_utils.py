import numpy as np

def predict(network, inputs):
    """
    Generates a prediction from the network based on the provided inputs.

    Args:
        network: Trained neural network structure.
        inputs: Input feature vector.

    Returns:
        Predicted class probabilities or outputs.
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
        z: Input to the activation function.
        activation: Type of activation function.

    Returns:
        Activated output.
    """
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'softmax':
        exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
        return exp_z / exp_z.sum(axis=0)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def interpret_prediction(output):
    """
    Interprets the network's output for chess scenarios, prioritizing Checkmate over Check.

    Args:
        output: Output vector from the neural network.

    Returns:
        Human-readable interpretation (e.g., checkmate, stalemate).
    """
    # Classes for prediction
    classes = [
        "Checkmate White",  # Index 0
        "Checkmate Black",  # Index 1
        "Stalemate",        # Index 2
        "Check White",      # Index 3
        "Check Black",      # Index 4
        "Nothing"           # Index 5
    ]

    # Sort indices by descending probability
    sorted_indices = np.argsort(output)[::-1]

    # Check priority
    # for idx in sorted_indices:
    #     if idx in [0, 1]:  # Checkmate cases
    #         return classes[idx]
    #     elif idx in [3, 4]:  # Check cases
    #         # Ensure no checkmate is present with higher probability
    #         if 0 not in sorted_indices[:2] and 1 not in sorted_indices[:2]:
    #             return classes[idx]
    # Debug print of every class and its probability
    # for idx in sorted_indices:
    #     print(f"{classes[idx]}: {output[idx]}")
    return classes[sorted_indices[0]]
