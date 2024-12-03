import numpy as np

PIECE_ENCODING = {
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,  # Black pieces
    'P': 1,  'N': 2,  'B': 3,  'R': 4,  'Q': 5,  'K': 6   # White pieces
}

def parse_piece_placement(piece_placement):
    board = []
    for rank in piece_placement.split('/'):
        row = []
        for char in rank:
            if char.isdigit():
                row.extend([0] * int(char))
            else:
                row.append(PIECE_ENCODING[char])
        board.append(row)
    return np.array(board).flatten()

def parse_active_color(active_color):
    return 1 if active_color == 'w' else -1

def parse_castling_availability(castling):
    availability = ['K', 'Q', 'k', 'q']
    return [1 if char in castling else 0 for char in availability]

def parse_en_passant(en_passant):
    if en_passant == '-':
        return 0
    file = ord(en_passant[0]) - ord('a')
    rank = int(en_passant[1]) - 1
    return rank * 8 + file

def parse_halfmove_clock(halfmove_clock):
    return int(halfmove_clock)

def parse_fullmove_number(fullmove_number):
    return int(fullmove_number)

def normalize_fen_vector(fen_vector):
    fen_vector = fen_vector.astype(np.float64)
    fen_vector[:64] /= 6
    if len(fen_vector) > 64:
        fen_vector[69] /= 63  # En passant
    if len(fen_vector) > 70:
        fen_vector[70] /= 100  # Halfmove clock
        fen_vector[71] /= 100  # Fullmove number
    return fen_vector

def parse_fen(fen, features):
    parts = fen.strip().split(' ')
    if len(parts) != 6:
        raise ValueError("Invalid FEN string: Must contain exactly 6 fields.")

    feature_map = {
        "piece_placement": parse_piece_placement(parts[0]),
        "active_color": parse_active_color(parts[1]),
        "castling_rights": parse_castling_availability(parts[2]),
        "en_passant": [parse_en_passant(parts[3])],
        "halfmove_clock": [parse_halfmove_clock(parts[4])],
        "fullmove_number": [parse_fullmove_number(parts[5])]
    }

    selected_features = [feature_map[feature] for feature in features]
    combined_vector = np.concatenate(selected_features)
    return normalize_fen_vector(combined_vector)

def preprocess_fen(fen_string, input_features):
    parts = fen_string.strip().split(' ')
    if len(parts) != 6:
        raise ValueError("Invalid FEN string: Must contain exactly 6 fields.")
    
    feature_map = {
        "piece_placement": np.array(parse_piece_placement(parts[0])),
        "active_color": np.array([parse_active_color(parts[1])]),
        "castling_rights": np.array(parse_castling_availability(parts[2])),
        "en_passant": np.array([parse_en_passant(parts[3])]),
        "halfmove_clock": np.array([parse_halfmove_clock(parts[4])]),
        "fullmove_number": np.array([parse_fullmove_number(parts[5])]),
    }
    
    extracted = np.concatenate([feature_map[feature] for feature in input_features if feature in feature_map])

    if extracted.size == 0:
        raise ValueError(f"No valid features extracted for FEN: {fen_string}")

    return extracted
