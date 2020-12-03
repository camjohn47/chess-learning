import string
import sys

import chess
import chess.polyglot

from chessai import ChessAI


# Method called at every turn to show the game's board, notation and available moves to the user.
def display(board):
    headers = f"\n{'Board': >10}{'Notation': >33}{'Your Available Moves': >43}"
    print(headers)
    moves = sorted([board.uci(move) for move in board.legal_moves])
    num_move_rows = int(len(moves) / moves_per_row)
    for row in reversed_rows:
        row_start, row_end = 8 * row, 8 * (row + 1)
        row_range = range(row_start, row_end)
        row_pieces = [board.piece_at(row) for row in row_range]
        row_symbols = " ".join(
            [chess.Piece.symbol(piece) if piece else "-" for piece in row_pieces]
        )
        row_label = str(abs(row) + 1)
        square_labels = []

        for column in columns:
            column_label = alphabet[column]
            square_label = column_label + row_label
            square_labels.append(square_label)

        labels = " ".join(square_labels)
        entry = f"{row_symbols}{labels: >35}"

        row = abs(7 - row)

        if row <= num_move_rows - 1:
            stop = min(len(moves), (row + 1) * moves_per_row)
            move_symbols = [str(move) for move in moves[row * moves_per_row : stop]]
            entry += f"{', '.join(move_symbols): >40}"

        print(entry)


# Determines how the game has ended.
def end_game(board):
    display(board)

    win = "Congratulations! You won."
    lost = "You have lost."
    if board.is_checkmate():
        print(f"\n{win if board.result()[0] == '1' else lost}")
    else:
        print("\nStalemate.")
    sys.exit()


def user_turn(board):
    prompt = "\nIt's your turn! Please submit one of the legal moves shown above.\n"
    user_move = chess.Move.from_uci(input(prompt))

    while user_move not in board.legal_moves:
        illegal_str = "Sorry, but that move is illegal."
        try_again_str = "Please try again with one of the legal moves shown above."
        user_move = chess.Move.from_uci(input(f"\n{illegal_str} {try_again_str}\n"))
        continue

    board.push(user_move)
    display(board)


def computer_turn(board):
    print("\nYour move has been successfully made. Thinking about next move...\n")
    black_move = ai.move_optimization(board, alpha, beta, depth)
    board.push(black_move)
    display(board)


if __name__ == "__main__":
    # Constants used in the displayed method to produce UI and visuals.
    alphabet = string.ascii_lowercase
    moves_per_row = 5
    reversed_rows = list(reversed(range(8)))
    columns = range(8)
    column_labels = "|".join(["a|", "b|", "c|", "d|", "e|", "f|", "g|", "h|"])
    headers = f"{'Board': >10}{'Notation': >33}{'Your Available Moves': >43}"

    # AI's game tree search parameters.
    beta = float("inf")
    alpha = -1 * beta
    depth = 4

    # Main script that interacts with the user, executes moves, and organizes the development of the game.
    board = chess.Board()

    # Path associated with the position cache, in which chess positions and their valuations are stored as they're calculated. 
    # NOTE: When changing models, the path should be changed as well. Without doing so, different chess positions can be 
    # calculated with different valuation functions in the same game.
    cache_path = "example_cache.data"
    model_path = 'model.data'
    ai = ChessAI(cache_path=cache_path,model_path=model_path)
    display(board)

    while not board.is_checkmate() and not board.is_stalemate():
        user_turn(board)
        computer_turn(board)

    print("")
    end_game(board)
