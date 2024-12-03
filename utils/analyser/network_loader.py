import json
import numpy as np

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
    with open(file_path, 'r') as file:
        combined_network = json.load(file, object_hook=numpy_decoder)
    return combined_network

def save_combined_network(networks, file_path):
    with open(file_path, 'w') as file:
        json.dump(networks, file, cls=NumpyEncoder, indent=4)
