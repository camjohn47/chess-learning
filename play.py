import string
import sys
from colorama import init
from termcolor import colored, cprint
import chess
import chess.pgn
from chessai import ChessAI
from chess_features import get_features
import math
import time

def display(board, ai_valuation=None):
    """
    Method called at every turn to show the game's board, notation, available
    moves and move history to the user.
    """

    print(headers)
    moves = sorted([board.uci(move) for move in board.legal_moves])
    num_move_rows = int(len(moves) / moves_per_row)
    display_rows = reversed_rows if user_side else rows

    for row_ind, row in enumerate(display_rows):
        # Build substring of UI output containing a row of the board's pieces,
        # and a row of the board's piece notation.

        row_squares = range(8 * row, 8 * (row + 1))
        row_pieces = [chess.Piece.symbol(board.piece_at(square)) 
                      if board.piece_at(square) else "-" for square in row_squares]

        row_colors = [piece_square_coloring(row_pieces[i], row_squares[i])
                      for i in range(8)]

        row_symbols = colored("", 'red', board_color).join(row_colors)

        row_notation = [alphabet[column] + row_labels[row] for column in columns]
        notation_colors = [notation_coloring(row_notation[i], row_squares[i])
                           for i in range(8)]

        notation = colored("", 'red', board_color).join(notation_colors)
        entry = f"{row_symbols: >20}     {notation: >30}     "

        # Add a substring containing a row of available user moves and move
        # history (previous moves of the game) to the UI output.
        row = invert_row(row) if user_side else row
        stop = min(len(moves), (row + 1) * moves_per_row) 
        move_symbols = [str(move) for move in moves[row * moves_per_row : stop]] 
        entry += f"{', '.join(move_symbols): <40}"
        hist_stop = min(len(move_hist), (row + 1) * moves_per_row) 
        move_hist_symbs = [str(move) for move in move_hist[row * moves_per_row : hist_stop]] 
        entry += f"{', '.join(move_hist_symbs): <40}"
        print(entry)

    # Print out extra move history rows, which is needed if it requires more than
    # 8 rows: (# moves / # moves per row) > 8.
    num_hist_rows = math.ceil(len(move_hist) / moves_per_row)
    extra_hist_rows = range(8, num_hist_rows)
    for row in extra_hist_rows:
        hist_stop = min(len(move_hist), (row + 1) * moves_per_row)
        move_hist_symbs = [str(move) for move in move_hist[(row) * moves_per_row : hist_stop]]
        entry = "".join([" " for space in range(106)])
        entry += f"{', '.join(move_hist_symbs): <40}"
        print(entry)

def get_pgn(board):
    """
    Prints a pgn encoding of the game that has been played. 
    """

    game = chess.pgn.Game()
    node = game
    for move in move_hist:
        node = node.add_variation(move)

    print(game)

def end_game(board, resign=False):
    """
    Determines how the game has ended. Concludes by stating how the game has ended, 
    and then providing the user the option to have a pgn encoding of the game. 
    """

    display(board)
    print((f"\n{win if board.result()[0] == '1' else loss}") if board.is_checkmate()
                   else (f"") if resign else stalemate_str)
    pgn_inputs = set(["pgn", "exit"])
    pgn_input  = input((f"You've resigned. If you would like a PGN encoding of the game, type and" 
                        f" enter 'pgn'. If not, enter 'exit'. "))

    while pgn_input not in pgn_inputs:
        pgn_input  = input(f"That's an invalid response. Please enter 'pgn' or 'exit'. ")

    if pgn_input == "pgn":
        get_pgn(board)

    sys.exit(0)

def user_prompt(ai_move, ai_valuation):
    """
    Determines which prompt is sent to the user at the beginning of its turn.
    """

    checkmate_coming = ai_valuation and ai_valuation == float('inf')

    if checkmate_coming:
        moves_til_checkmate = ai.moves_til_checkmate - 1
        winning_player = player_bool_to_name[not user_side] 
        preface = f"{winning_player} has checkmate in {moves_til_checkmate} moves. "
    else:
        prob = (100 if ai_valuation == -float('inf')
                   else (abs(int(round(ai_valuation, 2) * 100)) if ai_valuation
                   else int(round(ai.board_valuation(board), 2) * 100)))
        preface = (f"\n{opponent_name} has played {ai_move}."
                   f" It thinks your probability of winning is {prob}%. " if ai_move
                   else "")

    prompt = (preface +  "Enter an available move, or resign by entering 'resign': " if move_hist else
             f"You'll be playing as {user_name}. {opponent_name} will be played by {model_path}. Enter an available move: ")

    return prompt

def user_turn(board, ai_move=None, ai_valuation=None):
    """
    Manages the UI during user's turn. Once a valid move is received, it is 
    processed into the game's state and the function call ends. 
    """

    made_move = False
    prompt = user_prompt(ai_move, ai_valuation)

    while not made_move:
        user_entry = input(prompt)

        if user_entry == 'resign':
            end_game(board, resign=True)
        try:
            user_move = chess.Move.from_uci(user_entry)
        except ValueError:
            prompt = (f"\nYour move contains invalid squares. Try again with one"
                      f"of the available moves. ")
            continue

        if user_move not in board.legal_moves:
            illegal_str = "Sorry, but that move is illegal. "
            try_again_str = "Try again with one of the legal moves shown above. "
            prompt = f"\n{illegal_str} {try_again_str}\n"
            continue

        ai.make_move(board, user_move)
        move_hist.append(user_move)
        display(board)
        made_move = True

def ai_turn(board):
    """
    Executes the opponent's turn, which is played by the AI. Calculates an optimal
    opponent move using the AI initiated below, and then processes that move in 
    the game's internal state.
    """

    features = get_features(board)
    print(f"\nYour move has been successfully made. {opponent_name} is thinking about its next move...\n")
    ai.curr_tactical_val = ai.get_tactical_valuation(board, features)
    t_max = 30 if board.fullmove_number <= 2 else 60
    start = time.time()
    ai_move, ai_valuation = ai.iterative_depth_search(board, player=not user_side, stop_at_depth=True, t_max=t_max)

    print(f"Time: {time.time() - start}")
    ai.make_move(board, ai_move)
    move_hist.append(ai_move)
    ai.update()
    display(board)

    return ai_move, ai_valuation

if __name__ == "__main__":
    win, loss = "Congratulations, you won.", "You've been checkmated."
    resign_str = "You've resigned."
    stalemate_str = (f"The game has ended in a stalemate. There are no legal moves"
                    f"for either player.")
    row_labels = {row: str(row + 1) for row in range(8)}
    squares = range(64)
    invert_row = lambda row: abs(7-row)

    # Lambdas used for printing and coloring the UI boards' different pieces and squares during the game.
    is_square_even = lambda square: square % 2 == 0
    is_row_even = lambda square: int(square/8 % 2) == 0
    square_coloring = {square: 'on_red' if (is_square_even(square) and is_row_even(square))
                       or (not is_square_even(square) and not is_row_even(square))
                       else 'on_magenta' for square in squares}

    square_front_color = {square: 'red' if square_coloring[square] == 'on_red'
                          else 'magenta' for square in squares}

    piece_coloring = lambda piece, square: ('white' if piece in white_pieces else
                     'grey' if piece in black_pieces else square_front_color[square])

    piece_square_coloring = lambda piece, square: colored(' ' + piece + ' ',
                            piece_coloring(piece, square), square_coloring[square]) 
    notation_coloring = lambda notation, square: colored(f' {notation} ', 'grey',
                                                 square_coloring[square]) 
    
    # Parameters and data structures used for keeping track of moves and displaying the visuals. 
    white_pieces = set(['P', 'N', 'B', 'R', 'Q', 'K'])
    black_pieces = set(['p', 'n', 'b', 'r', 'q', 'k'])
    board_color = 'on_red'
    move_hist = []
    moves_per_row = 5
    rows = list(range(8))
    reversed_rows = list(reversed(rows))
    columns = range(8)
    alphabet = string.ascii_lowercase
    column_labels = "|".join(["a|", "b|", "c|", "d|", "e|", "f|", "g|", "h|"])
    headers =f"\n{'Board': >14}{'Notation': >35}{'Available Moves': >39}{'Move History': >38}"

    # Creating the ChessAI instance that generates opponent moves. The ChessAI
    # uses predictions made by the model found in <model_path> to evaluate chess
    # positions. Optional arg <valuation_path> can be used to load a previous
    # valuation cache. The file in valuation path should be of this form to work
    # properly: 
    # zobrist hash(position) -> valuation(position).

    model_path = 'chess_rfc.data'
    user_side_valid = False
    prompt = f"Would you like to play as White or Black? "
    user_side = False
    player_bool_to_name = {True: "White", False: "Black"}

    while not user_side_valid:
        requested_side = input(prompt)
        if requested_side in ["White", "white"]:
            user_side, user_side_valid = True, True
        elif requested_side in ["Black", "black"]:
            user_side, user_side_valid = False, True
        else:
            prompt = "Sorry, that's not a valid side to choose. Please enter 'White', or 'Black'. "

    user_name = player_bool_to_name[user_side]
    opponent_name = player_bool_to_name[not user_side]
    valuation_path = model_path.replace('.', '_model_valuations.')
    ai = ChessAI(model_path=model_path, valuation_path=valuation_path, reset_valuation=True)
    ai_move, ai_valuation = None, None
    min_depth = 4
    board = chess.Board()
    display(board)

    if user_side:
        while not board.is_game_over():
            user_turn(board, ai_move, ai_valuation)

            if board.is_game_over():
                end_game(board)

            ai_move, ai_valuation = ai_turn(board)

        end_game(board)
    else:
        while not board.is_game_over():
            ai_move, ai_valuation = ai_turn(board)

            if board.is_game_over():
                end_game(board)

            user_turn(board, ai_move, ai_valuation)

        end_game(board)
