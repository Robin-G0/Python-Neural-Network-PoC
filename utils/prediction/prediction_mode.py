from utils.prediction.prediction_utils import interpret_decision, predict
import sys
def predict_mode(networks, data):
    """
    Predicts outcomes for a dataset using all neural networks in sequence.

    Args:
        networks (dict): Dictionary of the neural networks.
        data (list): List of tuples (inputs, label), where inputs may be None if invalid.
    """
    print("Prediction mode...", file=sys.stderr)

    network_sequence = [
        "something_vs_nothing",
        "check_vs_stalemate",
        "checkmate_vs_check",
        "white_vs_black"
    ]

    for current_input, expected_label in data:
        if current_input is None:
            # print(f"Skipping invalid FEN. Expected label: {expected_label}", file=sys.stderr)
            # print("Invalid FEN")
            # continue
            print(f"Invalid FEN", file=sys.stderr)
            sys.exit(84)
        result_path = []

        print(f"\nExpected: {expected_label if expected_label else 'Unknown'}", file=sys.stderr)

        for network_name in network_sequence:
            network = networks[network_name]

            # Perform prediction
            try:
                output = predict(network, current_input)
                prediction = round(output[0])  # Binary prediction: 0 or 1
                result_path.append(prediction)
            except Exception as e:
                print(f"Error during prediction in {network_name}: {e}", file=sys.stderr)
                break

            print(f"Network: {network_name}, Output: {output}, Prediction: {prediction}", file=sys.stderr)

            # Stop if "Nothing" is predicted
            if network_name == "something_vs_nothing" and prediction == 0:
                print("Prediction stopped at Nothing.", file=sys.stderr)
                break

        # Interpret and display the final prediction
        final_prediction = interpret_decision(result_path)
        if expected_label:
            print(f"Final Prediction: {final_prediction} | Expected: {expected_label}", file=sys.stderr)
        else:
            print(f"Final Prediction: {final_prediction}", file=sys.stderr)
