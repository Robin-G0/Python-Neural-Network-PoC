import json
import numpy as np
import sys

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def numpy_decoder(dct):
    for key, value in dct.items():
        if isinstance(value, list):
            try:
                dct[key] = np.array(value)
            except ValueError:
                pass
    return dct

def load_combined_network(file_path):
    try:
        with open(file_path, 'r') as file:
            combined_network = json.load(file, object_hook=numpy_decoder)
        return combined_network
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(84)

def save_combined_network(networks, file_path):
    with open(file_path, 'w') as file:
        json.dump(networks, file, cls=NumpyEncoder, indent=4)
