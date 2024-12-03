from utils.prediction.prediction_utils import interpret_decision, predict
from utils.analyser.fen_parser import parse_fen
import sys

def parse_prediction_line(line):
    """
    Parses a line of input to extract the FEN string and optional label.

    Args:
        line (str): A line containing a FEN string, optionally followed by an expected label.

    Returns:
        tuple: (inputs, label) where inputs is the parsed FEN vector and label is the optional expected label.
    """
    parts = line.strip().split(" ")
    if len(parts) < 6:
        return None, None  # Invalid line

    # Extract FEN string
    fen_string = " ".join(parts[:6])

    # Parse expected label if present
    expected_label = " ".join(parts[6:]) if len(parts) > 6 else None

    try:
        inputs = parse_fen(fen_string)
        return inputs, expected_label
    except ValueError as e:
        print(f"Error parsing FEN: {fen_string}, {e}", file=sys.stderr)
        return None, None

def predict_mode(networks, data):
    """
    Predicts outcomes for a dataset using all neural networks in sequence.

    Args:
        networks (dict): Dictionary of the neural networks.
        data (list): List of tuples (inputs, label), where inputs are parsed FEN vectors, and label is optional.
    """
    print("Prediction mode...", file=sys.stderr)

    # Ordered list of neural networks
    network_sequence = [
        "something_vs_nothing",
        "check_vs_stalemate",
        "checkmate_vs_check",
        "white_vs_black"
    ]

    for current_input, expected_label in data:
        result_path = []  # Store intermediate results for debugging and final interpretation

        print(f"\nProcessing FEN: {expected_label if expected_label else 'Unknown'}", file=sys.stderr)

        for network_name in network_sequence:
            # Retrieve the current network
            network = networks[network_name]
            
            # Perform prediction
            output = predict(network, current_input)
            prediction = round(output[0])  # Binary prediction: 0 or 1
            result_path.append(prediction)

            print(f"Network: {network_name}, Output: {output}, Prediction: {prediction}", file=sys.stderr)

            # Break if "Nothing" is predicted
            if network_name == "something_vs_nothing" and prediction == 0:
                print("Prediction stopped at Nothing.", file=sys.stderr)
                break

        # Interpret and display the final prediction
        final_prediction = interpret_decision(result_path)
        if expected_label:
            print(f"Final Prediction: {final_prediction} | Expected: {expected_label}", file=sys.stderr)
        print(f"{final_prediction}")
