import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        return super().default(obj)

def numpy_decoder(dct):
    """
    Custom JSON decoder for NumPy arrays.
    Converts lists back into NumPy arrays when appropriate.
    """
    for key, value in dct.items():
        if isinstance(value, list):
            # Check if the list represents a NumPy array
            try:
                # Attempt to convert to a NumPy array
                dct[key] = np.array(value)
            except ValueError:
                # If conversion fails, leave it as is
                pass
    return dct

def load_network(file_path):
    """
    Loads a neural network from a JSON file, handling NumPy arrays.

    Args:
        file_path: Path to the network file.

    Returns:
        The loaded neural network.
    """
    with open(file_path, 'r') as file:
        return json.load(file, object_hook=numpy_decoder)

def save_network(network, file_path):
    """
    Saves a neural network to a JSON file, handling NumPy arrays.

    Args:
        network: Neural network structure.
        file_path: Path to save the network file.
    """
    with open(file_path, 'w') as file:
        json.dump(network, file, cls=NumpyEncoder, indent=4)
