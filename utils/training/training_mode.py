from utils.training.training_utils import get_label_map, filter_data_by_labels, train_network_multithreaded
from utils.analyser.network_loader import save_combined_network
import threading
import queue
import sys

def train_mode(networks, data, curve_enabled, save_file, spec):
    """
    Trains the neural networks using the provided data.

    Args:
        networks (dict): Dictionary of neural networks with their configurations.
        data (list): List of tuples containing inputs and labels.
        curve_enabled (bool): Whether to enable the learning curve.
        save_file (str): Path to save the trained networks.
        spec (int): Index of the network to train.
    """
    print("Training mode...", file=sys.stderr)
    label_maps = get_label_map()
    stop_flag = threading.Event()  # Shared stop flag

    if spec:
        network_indices = [spec - 1]
        print(f"Training only network {list(networks.keys())[spec - 1]}", file=sys.stderr)
    else:
        network_indices = range(len(networks))

    for idx in network_indices:
        network_name = list(networks.keys())[idx]
        network = networks[network_name]
        label_map = label_maps[network_name]

        training_data = [
            (inputs, label_map[label])
            for inputs, label in filter_data_by_labels(data, label_map)
        ]

        if not training_data:
            print(f"No valid data for network {network_name}. Skipping...", file=sys.stderr)
            continue

        curve = None
        updates_queue = queue.Queue()

        if curve_enabled:
            try:
                from bonus.learning_curve import LearningCurve
                curve = LearningCurve(save_file=save_file, networks=networks, stop_flag=stop_flag)
            except ImportError as e:
                print(f"Error: {e}, Cannot use +curve option.", file=sys.stderr)
                sys.exit(84)

        training_thread = threading.Thread(
            target=train_network_multithreaded,
            args=(network, training_data),
            kwargs={
                'learning_rate': 0.01,
                'epochs': 10,
                'batch_size': 128,
                'updates_queue': updates_queue,
                'stop_flag': stop_flag
            }
        )
        training_thread.start()

        try:
            while training_thread.is_alive():
                try:
                    loss, accuracy = updates_queue.get(timeout=0.1)
                    if curve_enabled and curve:
                        curve.update(loss, accuracy)
                except queue.Empty:
                    continue

            if curve_enabled and curve:
                curve.finalize()

            print(f"Training for network {network_name} completed.", file=sys.stderr)
            networks[network_name] = network  # Update the trained network
        except KeyboardInterrupt:
            print("\nTraining interrupted. Stopping...", file=sys.stderr)
            stop_flag.set()
            training_thread.join()
            sys.exit(84)

    save_combined_network(networks, save_file)
    print(f"Trained networks saved to {save_file}", file=sys.stderr)
