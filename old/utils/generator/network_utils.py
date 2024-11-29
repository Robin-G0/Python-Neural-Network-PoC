import json
import numpy as np
from pathlib import Path
import sys

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy arrays.
    
    This class extends the default JSON encoder to handle NumPy arrays.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        return super().default(obj)

def parse_config(config_path):
    """
    Parses a JSON configuration file.
    
    Args:
        config_path (str): The path to the configuration file.
    
    Returns:
        dict: The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file '{config_path}': {e}")
        sys.exit(1)

def initialize_weights(units, previous_units, activation):
    """
    Initializes the weights for a layer in a neural network.
    
    Args:
        units (int): The number of units in the layer.
        previous_units (int): The number of units in the previous layer.
        activation (str): The activation function for the layer.
        
    Returns:
        np.ndarray: The initialized weights for the layer.
    """
    if activation == "relu":
        return np.random.randn(units, previous_units) * np.sqrt(2 / previous_units)
    else:
        return np.random.randn(units, previous_units) * np.sqrt(1 / previous_units)

def generate_network(config, network_id, output_dir, config_path):
    """
    Generates a network and saves it as a file.
    
    Args:
        config (dict): The configuration dictionary for the network.
        network_id (int): The unique identifier for the network.
        output_dir (str): The directory to save the network file.
        config_path (str): The path to the configuration file.
        
    Returns:
        None
    """
    input_size = config.get('input_size')
    layers = config.get('layers')
    if not input_size or not layers:
        print("Error: Configuration file is missing required fields.")
        sys.exit(1)
    
    # Simulate creating a network with improved weight initialization
    np.random.seed(network_id)  # Ensure reproducibility for this network
    network = {
        "input_size": input_size,
        "layers": []
    }
    for i, layer in enumerate(layers):
        previous_units = input_size if i == 0 else layers[i-1]['units']
        layer_structure = {
            "weights": initialize_weights(layer['units'], previous_units, layer["activation"]).tolist(),
            "biases": np.zeros(layer['units']).tolist(),
            "activation": layer["activation"]
        }

        network["layers"].append(layer_structure)

    # Save the network with a .nn extension
    network_path = Path(output_dir) / f"{Path(config_path).stem}_{network_id}.nn"
    with open(network_path, 'w') as file:
        json.dump(network, file, cls=NumpyEncoder, indent=4)
    print(f"Network {network_id} saved to {network_path}")
