import numpy as np

PIECE_ENCODING = {
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,  # Black pieces
    'P': 1,  'N': 2,  'B': 3,  'R': 4,  'Q': 5,  'K': 6   # White pieces
}

def parse_piece_placement(piece_placement):
    """Converts the piece placement section of a FEN string to a 2D array."""
    board = []
    for rank in piece_placement.split('/'):
        row = []
        for char in rank:
            if char.isdigit():  # Empty squares
                row.extend([0] * int(char))
            else:  # Pieces
                row.append(PIECE_ENCODING[char])
        board.append(row)
    return np.array(board).flatten()

def parse_active_color(active_color):
    """Encodes the active color ('w' or 'b') as a binary value."""
    return 1 if active_color == 'w' else -1

def parse_castling_availability(castling):
    """Encodes castling availability as a binary vector."""
    availability = ['K', 'Q', 'k', 'q']
    return [1 if char in castling else 0 for char in availability]

def parse_en_passant(en_passant):
    """Encodes the en passant target square into a single numeric value."""
    if en_passant == '-':
        return 0
    # Convert the square (e.g., 'e3') into a numeric index
    file = ord(en_passant[0]) - ord('a')
    rank = int(en_passant[1]) - 1
    return rank * 8 + file

def parse_halfmove_clock(halfmove_clock):
    """Converts the halfmove clock to an integer."""
    return int(halfmove_clock)

def parse_fullmove_number(fullmove_number):
    """Converts the fullmove number to an integer."""
    return int(fullmove_number)

def normalize_fen_vector(fen_vector):
    """Normalize the FEN feature vector."""
    # Convert to float to avoid casting issues during normalization
    fen_vector = fen_vector.astype(np.float64)

    # Normalize board state (first 64 values)
    fen_vector[:64] /= 6
    # Normalize en passant (70th value)
    fen_vector[69] /= 63
    # Normalize halfmove clock and fullmove number (71st and 72nd values)
    fen_vector[70] /= 100
    fen_vector[71] /= 100
    return fen_vector

def parse_fen(fen):
    """Parses a FEN string into a normalized feature vector."""
    parts = fen.strip().split(' ')
    if len(parts) != 6:
        raise ValueError("Invalid FEN string: Must contain exactly 6 fields.")

    piece_placement = parse_piece_placement(parts[0])
    active_color = parse_active_color(parts[1])
    castling_availability = parse_castling_availability(parts[2])
    en_passant = parse_en_passant(parts[3])
    halfmove_clock = parse_halfmove_clock(parts[4])
    fullmove_number = parse_fullmove_number(parts[5])

    # Combine all features
    fen_vector = np.concatenate([
        piece_placement,
        [active_color],
        castling_availability,
        [en_passant],
        [halfmove_clock],
        [fullmove_number]
    ])

    # Normalize the feature vector
    return normalize_fen_vector(fen_vector)

