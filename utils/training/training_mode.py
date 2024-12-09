from utils.training.training_utils import (
    get_label_map, filter_data_by_labels, train_network_multithreaded, encode_one_hot
)
import threading
import queue
import sys

def train_mode(networks, data, curve_enabled, save_file):
    print("Training mode...", file=sys.stderr)
    label_maps = get_label_map()
    stop_flag = threading.Event()  # Shared stop flag

    for network_name, network in networks.items():
        if network_name not in label_maps:
            print(f"No label map for network {network_name}. Skipping...", file=sys.stderr)
            continue
        
        label_map = label_maps[network_name]

        # Gestion des labels encodés
        if network_name == "check_checkmate_stalemate":
            labels = [label_map[label] for _, label in filter_data_by_labels(data, label_map)]
            encoded_labels = encode_one_hot(labels, num_classes=3)
            training_data = [(inputs, label) for (inputs, _), label in zip(data, encoded_labels)]
        else:
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
                'learning_rate': 0.001,
                'epochs': 5,
                'batch_size': 500,
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
