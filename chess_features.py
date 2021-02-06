import numpy as np
import chess
from chess import square_file, square_rank, square_distance
from collections import Counter
from itertools import chain
from functools import reduce
import math

# True is white, False is black. 
players = [True, False]
white_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
black_pieces = ['p', 'n', 'b', 'r', 'q', 'k']
pieces = white_pieces + black_pieces
piece_type_to_index = {pieces[i]: i for i in range(len(pieces))}

# Pawn files that the pawn_development method considers.
center_rows = [chess.SquareSet([chess.C4, chess.D4, chess.E4, chess.F4]),
			   chess.SquareSet([chess.C5, chess.D5, chess.E5, chess.F5])]
center_squares = center_rows[0].union(center_rows[1])

# Approximate expected value of each piece type. Mostly used as a heuristic for 
# quickly and statically identifying blunders and bad moves. 
white_piece_vals = {'P': 1, 'N':3, 'B':3.3, 'R':4.2, 'Q':9, 'K':float('inf')}
black_piece_vals = {piece.lower(): -val for piece, val in white_piece_vals.items()}
piece_vals = dict(white_piece_vals, **black_piece_vals)
piece_vals['-'] = 0

# Kingside defensive information. 
white_kingside = [chess.F1, chess.F2, chess.F3, chess.G1, chess.G2, chess.G3,
				  chess.H1, chess.H2, chess.H3]
black_kingside = [chess.F6, chess.F7, chess.F8, chess.G6, chess.G7, chess.G8,
				  chess.H6, chess.H7, chess.H8]
white_kingside = chess.SquareSet(white_kingside)
black_kingside = chess.SquareSet(black_kingside)
opponentside = white_kingside | black_kingside
kingside_castles = set([chess.Move.from_uci('e1g1'), chess.Move.from_uci('e8g8')])
queenside_castles = set([chess.Move.from_uci('e1c1'), chess.Move.from_uci('e8c8')])

# Defining some key square sets of each player's side whose control is often
# a determining factor in the outcome of a chess game. 
player_to_kingside = {True: white_kingside, False: black_kingside}
whiteside = chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3)
blackside = chess.SquareSet(chess.BB_RANK_6 | chess.BB_RANK_7 | chess.BB_RANK_8)
player_to_side = {True: whiteside, False:blackside}
counted_pieces = range(1,6)
light_squares, dark_squares = chess.BB_LIGHT_SQUARES, chess.BB_DARK_SQUARES 

# Lambda functions used for straightforward, one-line feature calculations. A chess board is almost always included as input (except is information map),
# and is sometimes accompanied by a player (bool) argument to define the player whose pieces will be considered.
information_map = lambda prob: -prob*math.log(prob)
get_piece_counts = lambda board, player: [len(board.pieces(piece, player))
									      for piece in counted_pieces]

get_pieces_squares = lambda board, player: [board.pieces(piece, player)
										    for piece in range(1,7)]

get_attacks = lambda board, active_squares: [board.attacks(square)
											 for square in active_squares]

is_legal = lambda board, move: board.is_legal(move)
get_square_mobility = lambda board, square: len(board.attacks(square))
player_pawn_delta = {True: 8, False:-8}

def get_square_neighbors(square):
	"""
	Arguments: 
	square (chess Square): Chess board square whose neighbors will be returned. 

	Returns:
	Chess squareset containing the square's neighboring squares. More rigorously, {square'| dist(square, square') <= 1}, 
	"""
	square_neighbors = chess.SquareSet([neighbor for neighbor in chess.SQUARES
									    if square_distance(square, neighbor) <= 1])

	return square_neighbors

square_neighbors = {}
for square in chess.SQUARES:
	neighbors = get_square_neighbors(square)
	square_neighbors[square] = neighbors

def get_attacks_from(board, player, from_square):
	"""
	Arguments: 
	board (chess Board): Chess position.
	player (string): Name of player's color. Either "White" or "Black". 
	from_square (chess Square): Starting square of the attack.  

	Returns:
	attacks (list): A list of opponent squares attacked by player from <from_square>. 
	"""

	opponent = not player
	attacks = [square for square in board.attacks(from_square)
			   if board.color_at(square) == opponent]

	return attacks

def get_attacks_on(board, player, to_square):
	"""
	New arguments:
	to_square (chess Square): end square on which attacks are calculated.

	Returns:
	List of opponent squares whose pieces are attacking player's <to_square>. 
	If <to_square> isn't occupied by player, returns an empty list.
	"""

	active_attacks = ([square for square in board.attackers(not player, to_square)]
					   if board.color_at(to_square) == player else [])

	return active_attacks

def is_square_attacked_by(board, player, square):
	"""
	Returns:
	Boolean indicating whether a <player> piece attacks <square>.
	"""

	attacked = len(board.attackers(player, square)) > 0

	return attacked

def get_square_piece(board, square):
	"""
	Returns:
	Typing (string) of the piece occupying <square>.
	"""

	piece = board.piece_at(square)

	if piece:
		piece = chess.Piece.symbol(piece)
	else:
		piece = '-'

	return piece

def get_square_player(board, square):
	piece = get_square_piece(board, square)
	player = (True if piece in white_piece_vals
			  else (False if piece in black_piece_vals else None))

	return player

def get_square_piece_index(board, square):
	"""
	Returns:
	Index (integer) of the piece type occupying <square>.
	"""

	piece_type = get_square_piece(board, square)
	piece_index = piece_type_to_index[piece_type]

	return piece_index

def get_square_val(board, square):
	"""
	Returns:
	Float representing expected value of the piece type occupying <square>.
	"""

	piece = get_square_piece(board, square)
	val = piece_vals[piece]

	return val

def get_mobility_entropy(board, player):
	"""
	Returns:
	Entropy (float) of player's piece mobilities. More precisely, the following:
	output = entropy(Mob_i), where Mob_i is a random variable = total # of moves 
	available to <player> using ith piece type. Higher values indicate that the
	player's different pieces are equally represented in available moves 
	(generally good). Lower values indicate that the player's moves are 
	concentrated on a subset of pieces (generally bad).
	"""

	piece_mobilities = get_piece_mobilities(board, player)
	total_mobility = sum(piece_mobilities)
	start_probs = piece_mobilities / total_mobility
	mobility_entropy = sum(map(information_map, start_probs))

	return mobility_entropy

def get_defensive_feats(board, player, player_squares, opponent_squares):
	"""
	Arguments:
	player (bool): Denotes the player whose defensive features will be calculated.
	player_squares (chess SquareSet): Set of player's occupied squares.
	opponent_squares (chess Squareset): Set of opponent's occupied squares.

	Returns:
	Kingside attackers count, kingside defenders count, # of overworked kingside
	squares, # of attacked kingside squares, # defended kingside squares, 
	# defending pieces next to king, # opponent attacking pieces next to king.
	"""

	king_square = board.pieces(6, player).pop()
	kingside_attackers = [board.attackers(not player, square) 
								 for square in square_neighbors[king_square]]
	num_kingside_attackers = len(reduce(chess.SquareSet.union, kingside_attackers))
	kingside_defenders = [board.attackers(player, square)
								 for square in square_neighbors[king_square]]
	num_kingside_defenders = len(reduce(chess.SquareSet.union, kingside_defenders))
	num_kingside_overworked = sum([len(x) > len(y) for x,y in
								   zip(kingside_attackers, kingside_defenders)])
	num_kingside_attacked = sum([bool(x) for x in kingside_attackers])
	num_kingside_defended = sum([bool(x) for x in kingside_defenders])
	defenders_by_king = len(player_squares & square_neighbors[king_square])
	attackers_by_king = len(opponent_squares & square_neighbors[king_square])

	defensive_feats = [num_kingside_attackers, num_kingside_defenders,
	 				  num_kingside_overworked, num_kingside_attacked, 
	 				  num_kingside_defended, defenders_by_king, attackers_by_king]

	return defensive_feats

def get_square_control(board, square):
	"""
	Returns:
	The difference between the number of White and Black attacks on the square.
	If square control > 0, then White has more attacks than Black; if < 0, then 
	Black has more attacks. 
	"""

	num_white_attacks = len(board.attackers(True, square))
	num_black_attacks = len(board.attackers(False, square))
	square_control = num_white_attacks - num_black_attacks

	return square_control

def get_center_metrics(board, player_squares):
	"""
	Returns:
	Metrics ([int]) measuring each player's control of the center: 

	[# White center pieces, # Black center pieces, center square_1 control,...
	center square_N control],

	where center square_i control is an int indicating how many more attacks White
	has on center square_i relative to Black. 
	"""

	white_squares, black_squares = player_squares
	white_attackers = chess.SquareSet(chain(*[board.attackers(True, square)
											  for square in center_squares]))
	black_attackers = chess.SquareSet(chain(*[board.attackers(False, square)
											  for square in center_squares]))
	white_center_pieces = (white_squares & center_squares) | white_attackers
	black_center_pieces = (black_squares & center_squares) | black_attackers
	num_white_center_pieces = len(white_center_pieces)
	num_black_center_pieces =  len(black_center_pieces)
	center_control = [get_square_control(board, square) for square in center_squares]
	center_metrics = [num_white_center_pieces, num_black_center_pieces, *center_control]

	return center_metrics

def get_opponentside_metrics(board, white_squares, black_squares):
	"""
	Arguments: 
	white_squares: List ([chess Square]) containing White's active squares. 
	black_squares: List ([chess Square]) containing Black's active squares. 

	Returns:
	Metrics ([int]) measuring each player's control of squares on opponent's side:
	
	[# White pieces attacking Black side, # Black pieces attacking White side].
	"""

	white_attackers = chess.SquareSet(chain(*[board.attackers(True, square)
											  for square in player_to_side[False]]))
	black_attackers = chess.SquareSet(chain(*[board.attackers(False, square)
											  for square in player_to_side[True]]))
	opponentside_metrics = [len(white_attackers), len(black_attackers)]

	return opponentside_metrics

counted_pieces = range(1,6)
def count_player_pieces(board, player):
	"""
	Returns list containing player's piece type counts: [# pawns, ..., # queens].
	"""

	piece_counts = [len(board.pieces(piece, player)) for piece in counted_pieces]

	return piece_counts

def count_pieces(board):
	"""
	Returns concatenation of White and Black's piece counts:

	[# White pawns, ..., # White queens, # Black pawns, ..., # Black queens].
	"""

	white_piece_counts = count_player_pieces(board, True)
	black_piece_counts = count_player_pieces(board, False)
	piece_counts = white_piece_counts + black_piece_counts

	return piece_counts

def get_player_squares(board, player_pieces_squares):
	"""
	Returns set of all chess squares occupied on the board by player.
	"""

	pieces_squares = pieces_squares if pieces_squares else get_pieces_squares(board, player)
	player_squares = reduce(chess.SquareSet.union, pieces_squares)

	return player_squares

def get_immobile_piece_counts(board, white_squares=None, black_squares=None):
	"""
	Returns the net (White - Black) # of each piece type that is immobile:

	[# White immobile pawns - # Black immobile pawns, ...].

	Piece immobility is undesirable, especially for more valuable pieces like queens
	and kings. Immobile pieces have no legal moves, and are therefore trapped. 
	This reduces their positional value and can cause defensive/tactical risks as well.
	For ex, an immobile king is often very problematic for its owning player, and can 
	quickly lead to checkmate if not handled properly.
	"""

	white_squares = white_squares if white_squares else get_player_squares(board, True)
	black_squares = black_squares if black_squares else get_player_squares(board, False)
	squares = white_squares | black_squares
	immobile_piece_inds = [get_square_piece_index(square)
						   for square in squares if not board.attacks(square)]
	immobile_piece_counts = [0 for piece in range(12)]

	for piece_ind in immobile_piece_inds:
		immobile_piece_counts[piece_ind] += 1

	return immobile_piece_counts

def get_piece_mobilities(board, player=True, pieces_squares=None):
	"""
	Arguments:
	board (chess Board): Position to be analyzed.
	pieces_squares (nested chess Square list): list of each piece type's active squares for a player. 

	Returns: Float list containing the total # of moves available to the player for each piece type:

	[total pawn mobility(player), ..., total queen mobility(player)]
	"""

	pieces_squares = pieces_squares if pieces_squares else get_pieces_squares(board, player)
	granular_piece_mobilities = [sum([get_square_mobility(board, square)	
									  for square in piece_squares])
									  for piece_squares in pieces_squares]
	piece_mobilities = [sum(piece_mobilities)
					    for piece_mobilities in granular_piece_mobilities]

	return piece_mobilities

def get_mobility_metrics(board, player_squares, player_pieces_squares):
	"""
	Arguments:
	board (chess Board): Position to be analyzed.
	player_squares ([[chess Square]]): Nested list containing chess squares 
	occupied by each player: [white chess Squares, black chess Squares]
	pieces_squares ([[[chess Square]]]): Double-nested list containing chess squares 
	of each piece type for each player: [player piece squares for player in players].

	Returns: Float list containing the total # of moves available to each player,
	the total mobilities of each player's piece type, and immobile counts of each
	player's piece type.

	[total pawn mobility(player), ..., total queen mobility(player)]
	"""

	mobility_metrics = []
	player = True
	for squares, pieces_squares in zip(player_squares, player_pieces_squares):
		mobility = (lambda board, square: len(board.attacks(square) - squares)
										  if board.attacks(square) else 0)
		piece_mobilities = [[mobility(board, square) for square in piece_squares]
							 for piece_squares in pieces_squares]
		piece_total_mobilities = [sum(mobs) for mobs in piece_mobilities]
		total_mobility = sum(piece_total_mobilities)
		pawn_squares = pieces_squares[0]
		immobile_pawn_count = sum([(pawn_square + player_pawn_delta[player])
								    in squares for pawn_square in pawn_squares]) 
		immobile_counts = ([immobile_pawn_count] + 
		 				  [piece_mobilities[i].count(0) for i in range(1,6)])
		mobility_metrics.append([piece_total_mobilities, total_mobility, immobile_counts])
		player = False

	return mobility_metrics

def get_player_attacks(board, player_squares, opponent_squares):
	"""
	New arguments:
	player_squares (chess Square list): List of a player's active squares.
	opponent_squares (chess Square list): List of opponent's active squares.

	Returns:
	Integer count of all attacks from a player square to an opponent square.
	"""

	piece_attacks = [len(opponent_squares & board.attacks(square))
					 for square in player_squares]
	num_attacks = sum(piece_attacks)

	return num_attacks

def get_attacks(board, player_squares):
	"""
	Returns an np int array [White attack count, Black attack count]. Attacks
	are defined as moves which attack an opponent piece.
	"""

	white_attacks = get_player_attacks(board, player_squares[0], player_squares[1])
	black_attacks = get_player_attacks(board, player_squares[1], player_squares[0])
	attacks = [white_attacks, black_attacks]

	return attacks

def get_player_defends(board, player_squares):
	"""
	Returns integer count of the player's moves that defend a player piece.
	"""

	piece_defends = [len(player_squares & board.attacks(square))
					 for square in player_squares]
	num_defends = sum(piece_defends)

	return num_defends

def get_defends(board, player_squares):
	"""
	Returns an np int array containing the defensive moves count of each player:

	[# of white defends, # of black defends]. 
	"""

	defends = [get_player_defends(board, squares) for squares in player_squares]

	return defends

def get_pawn_metrics(player_pawn_squares):
	"""
	Returns a 16 x 1 np int array of each player's pawn count by file:

	[a-file White pawn count, ...., h-file White pawn count, a-file Black pawn
	count,..., h-file Black pawn count]
	"""

	pawn_metrics = []
	for pawn_squares in player_pawn_squares:
		pawn_files = map(chess.square_file, pawn_squares)
		pawn_file_counter = Counter(pawn_files)
		pawn_file_counts =  [pawn_file_counter[file] if file in pawn_file_counter
							 else 0 for file in range(8)]
		pawn_metrics += pawn_file_counts
	
	return pawn_metrics

def get_move_exchange(board, move, player):
	"""
	Computationally cheap (relative to enhanced move sorting it brings) heuristic
	used for sorting moves. Approximates the net material exchange of a move that
	attacks an opponent square. The idea is to quickly identify blunders and good/bad
	captures without having to search the position (statically). Assumes the
	opponent will capture with least valuable defending piece, if available. As
	the exchange depth exceeds 2, the approximation error grows quickly.

	Returns:
	Float representing the approximate, immediate (depth = 2) material exchange
	if player makes this capturing move.
	"""

	opponent = not player
	from_square, to_square = move.from_square, move.to_square
	to_square_val = get_square_val(board, to_square)
	move_exchange = -to_square_val 

	if is_square_attacked_by(board, opponent, to_square):
		attacker_vals = [get_square_val(board, square)
						 for square in board.attackers(not player, to_square)]
		worst_attacker_val = min(attacker_vals) if player else max(attacker_vals)
		from_square_val = get_square_val(board, from_square)
		board.push(move)
		will_attack = True

		if len(board.attackers(player, to_square)) > 1:
			will_attack = worst_attacker_val <= -from_square_val
			move_exchange = (move_exchange - from_square_val - worst_attacker_val
						    if will_attack else move_exchange)
		else:
			move_exchange = move_exchange - from_square_val

		board.pop()

	return move_exchange

def get_gamestage(board):
	"""
	A simple heuristic used to classify a position into opening, middle, and endgame. Motivation is to avoid
	using exact turn numbers, as this introduces noise into a highly selective model's learning process (like rfc, dtc).
	"""

	turn = board.fullmove_number
	stage = turn / 10

	return stage

def get_square_color_metrics(player_pieces_squares):
	"""
	Arguments: 
	pieces_squares (nested square list): List containing each piece type's list of squares:
	[pawn squares, knight squares, bishop squares, rook squares, queen square, king square]

	Returns: 
	A list of metrics indicating which player's colored bishops are active and that player's pawn counts
	on light vs. dark squares.
	"""

	square_color_metrics = []
	for pieces_squares in player_pieces_squares:
		pawn_squares, bishop_squares = pieces_squares[0], pieces_squares[2]
		light_bishop = int(len(bishop_squares & light_squares) > 0)
		dark_bishop = int(len(bishop_squares & dark_squares) > 0)
		light_pawn_squares = len(pawn_squares & light_squares)
		dark_pawn_squares = len(pawn_squares) - light_pawn_squares
		square_color_metrics.append([light_bishop, dark_bishop,
									 light_pawn_squares, dark_pawn_squares])

	return square_color_metrics


def get_features(board, has_castled=np.zeros((2))):
	"""
	IMPORTANT: This method is used by the ChessPipeline class to build features
	for training models. It is also used by the AI class to build the features 
	with which it represents chess positions. The whole system heavily depends on
	this method, so changes to it should be done dilligently. 

	Arguments:
	board (chess Board): Board representing chess position to be analyzed.
	has_castled (np int array): [white castle state, black castle state], where
	castle state = 0, 1, 2 -> not castled, kingside castled, queenside castled, respectively.

	Returns:
	Np array containing all features used for model representation of the chess
	position. There are currently 96. You can find out more detailed infromation 
	about each feature in the docstring of the method from which it is calculated.
	"""

	gamestage = get_gamestage(board)
	active_player = int(board.turn)
	check = board.is_check()

	player_piece_counts = [get_piece_counts(board, player) for player in players]
	player_piece_totals = [sum(piece_counts) for piece_counts in player_piece_counts]
	total_pieces = sum(player_piece_totals)

	player_pieces_squares = [get_pieces_squares(board, player) for player in players]
	player_squares = [reduce(chess.SquareSet.union, pieces_squares)
					  for pieces_squares in player_pieces_squares]
	player_mobility_metrics = get_mobility_metrics(board, player_squares, player_pieces_squares)
	player_attacks = get_attacks(board, player_squares)
	center_metrics = get_center_metrics(board, player_squares)
	white_squares, black_squares = player_squares
	opponentside_metrics = get_opponentside_metrics(board, white_squares, black_squares)

	player_squares_w, player_squares_b = player_squares
	white_defensive_feats = get_defensive_feats(board, True, player_squares_w, player_squares_b)
	black_defensive_feats = get_defensive_feats(board, False, player_squares_b, player_squares_w)
	player_pawn_squares = [pieces_squares[0] for pieces_squares in player_pieces_squares]
	pawn_metrics = get_pawn_metrics(player_pawn_squares)
	pawn_counts_by_file = pawn_metrics
	player_defends = get_defends(board, player_squares)
	white_color_metrics, black_color_metrics = get_square_color_metrics(player_pieces_squares)

	total_player_mobilities = [mobility_metrics[1]
							   for mobility_metrics in player_mobility_metrics]
	white_attacks, black_attacks = player_attacks
	white_defends, black_defends = player_defends
	white_piece_counts, black_piece_counts = player_piece_counts
	player_piece_mobilities = [mobility_metrics[0] 
							   for mobility_metrics in player_mobility_metrics]
	white_piece_mobilities, black_piece_mobilities = player_piece_mobilities
	immobile_piece_counts = list(chain(*[mobility_metrics[2]
								 for mobility_metrics in player_mobility_metrics]))

	features = [gamestage, active_player, check, total_pieces,
	            *total_player_mobilities, *player_attacks, *player_defends,
	            *white_piece_counts, *black_piece_counts, *white_piece_mobilities,
	            *black_piece_mobilities, *immobile_piece_counts, *center_metrics,
	            *opponentside_metrics, *white_defensive_feats, *black_defensive_feats,
				*has_castled, *pawn_counts_by_file, *white_color_metrics, 
				*black_color_metrics]

	features = np.array(features)

	return features

def get_castling_type(move):
	castling_type = (1 if move in kingside_castles
				    else (2 if move in queenside_castles else 0))
	
	return castling_type



