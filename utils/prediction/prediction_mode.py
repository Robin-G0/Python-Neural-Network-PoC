import sys
import numpy as np
from utils.analyser.fen_parser import preprocess_fen
from utils.prediction.prediction_utils import predict, interpret_decision
from utils.training.training_utils import print_progress_bar

def predict_mode(networks, input_file):
    """
    Predicts chessboard states using a chain of neural networks.

    Args:
        networks (dict): Dictionary of neural networks with their configurations.
        input_file (str): Path to the input file containing FEN strings and optional labels.
    """
    print(f"Predicting chessboard states from: {input_file}", file=sys.stderr)

    with open(input_file, 'r') as file:
        fen_lines = [line.strip() for line in file if line.strip()]

    predictions = []
    total_labels = 0
    correct_predictions = 0

    for idx, line in enumerate(fen_lines):
        try:
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
                # Check vs Stalemate
                inputs = preprocess_fen(fen, networks['check_vs_stalemate']['input_features'])
                check_result = predict(networks['check_vs_stalemate'], inputs)
                decision = int(check_result > 0.5)
                current_prediction.append(decision)

                if decision == 1:
                    # Checkmate vs Check
                    inputs = preprocess_fen(fen, networks['checkmate_vs_check']['input_features'])
                    checkmate_result = predict(networks['checkmate_vs_check'], inputs)
                    decision = int(checkmate_result > 0.5)
                    current_prediction.append(decision)

            # White vs Black
            inputs = preprocess_fen(fen, networks['white_vs_black']['input_features'])
            color_result = predict(networks['white_vs_black'], inputs)
            decision = int(color_result > 0.5)
            current_prediction.append(decision)

            # print(f"Prediction: {current_prediction}", file=sys.stderr) # to debug the prediction chain

            interpreted_result = interpret_decision(current_prediction)
            predictions.append((fen, interpreted_result, labels))

            print_progress_bar(idx + 1, len(fen_lines), prefix='Progress:', suffix='Complete')

        except Exception as e:
            print(f"Error processing FEN {idx + 1}: {e}", file=sys.stderr)

    # Display results
    for fen, result, labels in predictions:
        print(f"FEN: {fen}", file=sys.stderr)
        print("Prediction:", file=sys.stderr)
        print(result)
        if labels:
            print("Expected Label:", ' '.join(labels), file=sys.stderr)
            total_labels += 1
            if result in labels:
                correct_predictions += 1
        print("", file=sys.stderr)

    if total_labels > 0:
        accuracy = correct_predictions / total_labels * 100
        print(f"Accuracy: {accuracy:.2f}%", file=sys.stderr)
    else:
        print("Accuracy not available for this session.", file=sys.stderr)