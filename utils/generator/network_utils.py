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
            return obj.tolist()
        return super().default(obj)

def parse_config(config_path):
    """
    Parses a JSON configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        The parsed configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(84)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file '{config_path}': {e}", file=sys.stderr)
        sys.exit(84)

def initialize_weights(units, previous_units, activation):
    """
    Initializes the weights of a neural network layer.

    Args:
        units: Number of units in the layer.
        previous_units: Number of units in the previous layer.
        activation: Activation function of the layer.

    Returns:
        NumPy array of initialized weights.
    """
    if activation == "relu":
        return np.random.randn(units, previous_units) * np.sqrt(2 / previous_units)
    else :
        return np.random.randn(units, previous_units) * np.sqrt(1 / previous_units)

def generate_combined_network(config):
    """
    Generates a dictionary containing networks based on the provided configuration.
    
    Args:
        config (dict): The configuration dictionary with a 'networks' key containing individual network definitions.
    
    Returns:
        dict: A dictionary with networks labeled by their respective tasks.
    """
    if "networks" not in config or not isinstance(config["networks"], list):
        print("Error: Configuration file is missing the 'networks' key or it's not a list.", file=sys.stderr)
        sys.exit(84)

    combined_networks = {}
    for network_conf in config["networks"]:
        name = network_conf.get("name")
        input_features = network_conf.get("input_features", [])
        input_size = network_conf.get("input_size")
        layers = network_conf.get("layers")

        if not name or not input_features or not input_size or not layers:
            print(f"Error: Missing required fields in network configuration: {network_conf}", file=sys.stderr)
            sys.exit(84)

        network = {
            "input_features": input_features,
            "input_size": input_size,
            "layers": []
        }

        for i, layer in enumerate(layers):
            previous_units = input_size if i == 0 else layers[i - 1]["units"]
            layer_structure = {
                "weights": initialize_weights(layer["units"], previous_units, layer["activation"]).tolist(),
                "biases": np.zeros(layer["units"]).tolist(),
                "activation": layer["activation"]
            }
            network["layers"].append(layer_structure)

        combined_networks[name] = network

    return combined_networks

def generate_multiple_combined_files(config, num_files, output_dir, config_path):
    """
    Generates multiple combined networks and saves them to separate files.

    Args:
        config: Configuration dictionary.
        num_files: Number of files to generate.
        output_dir: Directory to save the files.
        config_path: Path to the configuration file.

    Returns:
        None
    """
    for file_id in range(1, num_files + 1):
        np.random.seed(file_id)  # Ensure reproducibility per file
        combined_networks = generate_combined_network(config)
        file_name = Path(output_dir) / f"{Path(config_path).stem}_{file_id}.nn"
        with open(file_name, 'w') as file:
            json.dump(combined_networks, file, cls=NumpyEncoder, indent=4)
        print(f"File {file_name} saved with networks labeled by tasks.", file=sys.stderr)
