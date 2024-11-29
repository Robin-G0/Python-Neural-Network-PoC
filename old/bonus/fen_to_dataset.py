#!/usr/bin/env python3
import chess
import chess.engine
import csv

def determine_label(fen):
    board = chess.Board(fen)
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return "Checkmate Black"
        else:
            return "Checkmate White"
    elif board.is_stalemate():
        return "Stalemate"
    elif board.is_check():
        if board.turn == chess.WHITE:
            return "Check Black"
        else:
            return "Check White"
    else:
        return "Nothing"

def process_fen_file(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        csv_reader = csv.reader((line.replace('"', '') for line in infile))
        csv_writer = csv.writer(outfile, delimiter=' ')

        next(csv_reader)

        for row in csv_reader:
            fen_with_eval = row[0]
            fen = fen_with_eval.split(",")[0]  # Extract FEN string
            label = determine_label(fen)
            csv_writer.writerow([fen, label])

# Specify input and output file paths
input_file = "datasets/originals/chessData.csv"
output_file = "processed_fen_dataset.txt"

# Process the dataset
process_fen_file(input_file, output_file)
