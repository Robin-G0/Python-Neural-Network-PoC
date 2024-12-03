import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy arrays.

    This class extends the default JSON encoder to handle NumPy arrays.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def numpy_decoder(dct):
    """
    Custom JSON decoder for NumPy arrays.
    Converts lists back into NumPy arrays when appropriate.

    Args:
        dct: JSON dictionary to decode.

    Returns:
        The decoded JSON dictionary.
    """
    for key, value in dct.items():
        if isinstance(value, list):
            try:
                dct[key] = np.array(value)
            except ValueError:
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

    Returns:
        None
    """
    with open(file_path, 'w') as file:
        json.dump(network, file, cls=NumpyEncoder, indent=4)
