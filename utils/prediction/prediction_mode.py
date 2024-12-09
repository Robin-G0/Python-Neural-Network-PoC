import sys
import numpy as np
from utils.analyser.fen_parser import preprocess_fen
from utils.prediction.prediction_utils import predict, interpret_decision
from utils.training.training_utils import print_progress_bar

def load_fen_lines(input_file):
    """Loads and processes FEN strings from the input file."""
    with open(input_file, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def process_fen_line(line, networks):
    """Processes a single FEN line and generates predictions."""
    parts = line.split(' ', maxsplit=6)
    fen = ' '.join(parts[:6])
    labels = parts[6:] if len(parts) > 6 else None

    current_prediction = []

    # Something vs Nothing
    inputs = preprocess_fen(fen, networks['something_vs_nothing']['input_features'])
    something_result = predict(networks['something_vs_nothing'], inputs)
    decision = int(something_result > 0.5)
    current_prediction.append(decision)

    if decision == 1:
        # Check-Checkmate-Stalemate Prediction
        inputs = preprocess_fen(fen, networks['check_checkmate_stalemate']['input_features'])
        result = predict(networks['check_checkmate_stalemate'], inputs)
        decision = np.argmax(result)
        current_prediction.append(decision)

    # White vs Black Prediction
    inputs = preprocess_fen(fen, networks['white_vs_black']['input_features'])
    color_result = predict(networks['white_vs_black'], inputs)
    decision = int(color_result > 0.5)
    current_prediction.append(decision)

    interpreted_result = interpret_decision(current_prediction)
    return fen, interpreted_result, labels

def display_results(predictions):
    """
    Displays the predictions and calculates accuracy if labels are provided.
    """
    total_labels = 0
    correct_predictions = 0

    for fen, result, labels in predictions:
        print(f"\nFEN: {fen}", file=sys.stderr)
        print("Prediction:", result, file=sys.stderr)
        
        if labels:
            print("Expected:", labels[0], file=sys.stderr)
            total_labels += 1
            if result == labels[0]:
                correct_predictions += 1
            else:
                print("Incorrect prediction!", file=sys.stderr)
        
        print("", file=sys.stderr)

    if total_labels > 0:
        accuracy = correct_predictions / total_labels * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}%", file=sys.stderr)
    else:
        print("Accuracy not available for this session.", file=sys.stderr)

def calculate_and_print_accuracy(predictions, networks_count):
    """
    Calculates and prints the accuracy for each network based on predictions and expected labels.

    Args:
        predictions (list): List of tuples (FEN, result, labels).
        networks_count (int): Number of networks in the chain.
    """
    correct_counts = [0] * networks_count  # Correct predictions for each network
    total_counts = [0] * networks_count    # Total number of predictions

    for _, result, labels in predictions:
        if not labels:
            continue

        # Get expected and actual decision paths
        expected_decision_path = interpret_label_path(labels[0])
        actual_decision_path = interpret_label_path(result)

        # Evaluate predictions network by network
        for network_idx in range(networks_count):
            if network_idx < len(expected_decision_path):
                total_counts[network_idx] += 1

                if (
                    network_idx < len(actual_decision_path)
                    and actual_decision_path[network_idx] == expected_decision_path[network_idx]
                ):
                    correct_counts[network_idx] += 1
                else:
                    break  # Stop if a network's prediction is wrong

    # Display accuracy results for each network
    print("\n--- Network Accuracy ---", file=sys.stderr)
    for network_idx in range(networks_count):
        if total_counts[network_idx] > 0:
            accuracy = (correct_counts[network_idx] / total_counts[network_idx]) * 100
            print(f"Network {network_idx + 1} Accuracy: {accuracy:.2f}% (Tests: {total_counts[network_idx]})", file=sys.stderr)
        else:
            print(f"Network {network_idx + 1} Accuracy: N/A (No tests)", file=sys.stderr)

def interpret_label_path(label):
    """
    Maps labels or results to their corresponding decision paths.
    """
    decision_map = {
        "Nothing": [0],
        "Stalemate": [1, 2],
        "Check Black": [1, 0, 0],
        "Check White": [1, 0, 1],
        "Checkmate Black": [1, 1, 0],
        "Checkmate White": [1, 1, 1],
    }
    return decision_map.get(label, [])

def predict_mode(networks, input_file):
    """
    Predicts chessboard states using a chain of neural networks.

    Args:
        networks (dict): Dictionary of neural networks with their configurations.
        input_file (str): Path to the input file containing FEN strings and optional labels.
    """
    print(f"Predicting chessboard states from: {input_file}", file=sys.stderr)
    fen_lines = load_fen_lines(input_file)
    predictions = []
    networks_count = len(networks)

    for idx, line in enumerate(fen_lines):
        try:
            fen, interpreted_result, labels = process_fen_line(line, networks)
            predictions.append((fen, interpreted_result, labels))
            print_progress_bar(idx + 1, len(fen_lines), prefix='Progress:', suffix='Complete')
        except Exception as e:
            print(f"Error processing FEN {idx + 1}: {e}", file=sys.stderr)

    display_results(predictions)
    calculate_and_print_accuracy(predictions, networks_count)
