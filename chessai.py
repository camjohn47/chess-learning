import os
import pickle
from chess import square_distance
import chess.polyglot
import numpy as np
import operator
import sys
from collections import Counter, defaultdict, OrderedDict
from chess_features import get_features, get_attacks_from, get_attacks_on
from chess_features import is_square_attacked_by, get_square_val, get_move_exchange
import time
from functools import partial
import math

class ChessAI():
	"""
	Class for generating automated chess play. Valuates chess positions using classifier predictions
	and heuristics to assess who is most likely to win (and by how much). Quickly searches over and analyzes
	different possible move sequences to find the optimal move for a given chess position.
	"""
	
	def __init__(self, model_path, valuation_path=None, reset_valuation=False):
		"""
		Arguments:
		model_path (string): Name of file from which ML classifier for predicting chess game winners is loaded.
		valuation_path (string): Pathname of the valuation cache where chess position valuations are stored.
		reset_valuation (bool): Determines whether the valuation cache will be started from scratch. 
		"""

		if not os.path.exists(model_path):
			print('ERROR: Model not found. Please check the model path again.')
			sys.exit(0)

		file = open(model_path,'rb')
		self.model = pickle.load(file)
		file.close()

		self.valuation_path = valuation_path
		self.valuation_cache = {}

		# If reset_valuation = True, then a new, empty valuation cache will be used for this AI instance. 
		# Whenever the model in a model_path learns from new data, or changes in any way, setting to True is a good idea. 
		if not reset_valuation and valuation_path:
			if os.path.exists(valuation_path):
				file = open(valuation_path,'rb')
				self.valuation_cache = pickle.load(file)
				file.close()

		# Constant parameters used to track pieces, players, and game.
		self.piece_indices = range(1,7)

		# True = White, False = Black.
		self.players = [True, False]
		self.has_castled = np.zeros((2))

		# Variables and data structures used to track move valuations and sort moves. 
		self.captures_cache = {}
		self.move_rank_cache = {}
		self.quiescence_squares_cache = {}
		self.move_hist = defaultdict(int)
		self.move_hist_scale = 1000
		self.tt = defaultdict(dict)
		self.max_sorting_depth = 3
		self.min_quiescence_depth = 4
		self.max_quiescence_sort_depth = 2
		self.move_hist_norm = 1
		self.move_hist_max = 1
		log_pawn_weight = 1
		log_deltas = [0.0, 3, 3.3, 4.5, 9, -0.98]
		self.logistic_coefs = np.array([log_pawn_weight + log_delta for log_delta in log_deltas])
		self.tactical_cache = {}
		self.harmonic_mean = lambda x,y: 2.0 / (1.0/x + 1.0/y)
		self.contraharmonic_mean = lambda x,y: (x**2 + y**2) / (x + y)
		self.piece_count_inds = range(10, 15)
		self.positional_weight, self.tactical_weight = 0.65, 0.35

		# Used to filter out capture moves that are very unlikely to raise alpha during quiescence search.
		# Requires value of captured piece to be within <max_capture_delta> of the last captured piece. 
		self.max_capture_delta = 2
		self.rank_cache = {}
		self.moves_til_checkmate = float('inf')

	def get_tactical_valuation(self, board, features):
		"""
		Returns a weighted sum of each player's piece counts, with each piece's count weighted by its average value. 
		Higher values are better for white; lower values are better for Black. 0 is evenly balanced. 

		Arguments:
		board (chess Board): Current chess position. 
		features (np array): Board's predictive features used for model predictions.
		"""

		piece_counts = [features[i] - features[i + 5] for i in self.piece_count_inds]
		net_pieces_mobility = np.sum(features[21:24] - features[27:30])
		tactical_feats = piece_counts + [net_pieces_mobility]
		tactical_hash = tuple(tactical_feats)

		if tactical_hash in self.tactical_cache:
			return self.tactical_cache[tactical_hash]

		player = board.turn
		tactical_val = 1.0 / (1.0 + math.exp(-np.dot(tactical_feats, self.logistic_coefs)))
		self.tactical_cache[tactical_hash] = tactical_val
		
		return tactical_val

	def get_positional_valuation(self, features):
		"""
		Returns ML model's probability of a white win using positional features.
		AI model must be an sklearn classifier capable of probabilistic predictions 
		(has a <predict_proba> method).

		Arguments:
		features (np array): numerical ML features built from the chess position to be evaluated.

		Returns:
		Float representing the likelihood of a white win.
		"""

		features = features.reshape(1, -1) 
		model_val = self.model.predict_proba(features)[0][1]

		return model_val

	def board_valuation(self, board, player="Black", negamax=False):
		"""
		Valuate a chess position in terms of white's advantage. A valuation cache is used for memoization to reduce runtime.
	
		Arguments:
		board (chess Board): board representing a position in a chess game.
		player (string): the player to move.

		Returns:
		Float representing white's advantage. If the position isn't tactical
		"""

		board_hash = chess.polyglot.zobrist_hash(board)

		if board_hash in self.valuation_cache:
			return self.valuation_cache[board_hash]

		elif board.is_checkmate():
			return -float('inf') if player else float('inf')

		features = get_features(board, self.has_castled)
		tactical_val = self.get_tactical_valuation(board, features)
		positional_val = self.get_positional_valuation(features)
		valuation = (self.positional_weight * positional_val) + (self.tactical_weight * tactical_val)
		self.valuation_cache[board_hash] = valuation

		return valuation

	def make_move(self, board, move):
		if board.is_castling(move):
			player = 0 if board.turn else 1
			self.has_castled[player] = 1

		board.push(move)

	def undo_move(self, board):
		move = board.pop()

		if board.is_castling(move):
			player = 0 if board.turn else 1
			self.has_castled[player] = 0

	def valuate_move(self, board, move, player):
		"""
		Temporarily make a move to estimate its short term value.

		Returns:
		Immediate valuation (float) of the position resulting from the move.
		"""

		self.make_move(board, move)
		valuation = self.board_valuation(board, not player)
		self.undo_move(board)

		return valuation

	def get_moves(self, board):
		return list(board.legal_moves)

	def is_move_check(self, board, move):
		board.push(move)
		check = True if board.is_check() else False
		board.pop()

		return check

	def is_move_checkmate(self, board, move):
		board.push(move)
		checkmate = True if board.is_checkmate() else False
		board.pop()

		return checkmate

	def is_tactical_move(self, board, move):
		tactical = board.is_capture(move) 

		return tactical

	def get_captures(self, board, prev_capture_value=0):
		"""
		Returns list of tactical (capture-based) moves from a chess board.
		"""

		is_good_capture = lambda move: board.is_capture(move) and abs(get_square_val(board, move.to_square)) + self.max_capture_delta >= abs(prev_capture_value)
		board_hash = chess.polyglot.zobrist_hash(board)

		if board_hash in self.captures_cache:
			return self.captures_cache[board_hash]

		good_captures = [response for response in self.get_moves(board) if is_good_capture(response)]
		self.captures_cache[board_hash] = good_captures

		return good_captures

	def rank_move(self, board, player, board_hash, move):
		"""
		Move ranking function used to sort moves according to expected value for the moving player. The more accurate
		this function is for predicting move values, the quicker move searches will be for finding optimal moves. This
		is because alpha-beta cutoffs occur faster and more often when the search starts with optimal moves.

		Returns a float representing an estimate of how good the move is, using immediate tactical exchanges, move history 
		from past alpha/beta searches, and heuristic valuations. If player is Black, lower ranks correspond to optimality.
		"""

		if board_hash in self.tt and move == self.tt[board_hash]['best_move'] or self.is_move_checkmate(board, move):
			rank = float('inf')

			return rank

		move_hash = str(chess.polyglot.zobrist_hash(board)) + str(player) + str(move)
		rank = 0

		if move_hash in self.move_rank_cache:
			rank = self.move_rank_cache[move_hash]
		else:
			is_attacked = board.is_capture(move) or is_square_attacked_by(board, not player, move.to_square)
			rank = 100 * (get_move_exchange(board, move, player) + 0.01)  if is_attacked else 0
			self.move_rank_cache[move_hash] = rank
		
		rank = -rank if not player else rank

		if move in self.move_hist:
			rank += self.move_hist[move]/self.move_hist_norm 
		elif rank == 0:
			player_delta = 1 if player else -1
			rank += player_delta * self.valuate_move(board, move, player)

		return rank

	def quiescence_search(self, board, alpha, beta, player, move, prev_capture_val=0, quiescence_squares=None, depth=0, max_depth=5):
		"""
		An extended search for evaluating unstable positions. The idea is to
		reduce the horizon effect by further exploring and analyzing unstable
		positions reached at the end of the alpha beta search (the following
		method). Unstable positions are those whose valuations are likely to
		change rapidly with additional depth. Such positions often include captures,
		new potential captures, and checks, which can be identified through simple
		heuristics. 

		An example is a position where White's queen just captured a pawn
		on a square defended by Black. Without an extended quiescence search, 
		this position is likely to be heavily misevaluated.

		Arguments:
		board (chess Board): The position to be analyzed.
		alpha (float): Current lower bound of the highest value White can attain.
		beta (float): Current upper bound of the lowest value Black can attain
		(low = better for Black).
		player (string): Active player.
		move (chess Move): The most recently made move.
		quiescence_squares: 
		depth (int): Depth of the current position being explored by quiescence
		search (with respect to position in alpha beta search that triggered the
		function call) .
		max_depth (int): Maximum depth of search. Needed to prevent massive
		runtimes in certain positions.

		Returns:
		best_val (float): A float representing an estimate of how favorable the
		best move for the given player is, with respect to White. 
		best_moves (chess Move list): Sequence of best minimax moves (first move
		is Black, second is White, ...) available for player.
		"""

		board_hash = chess.polyglot.zobrist_hash(board)
		height = max_depth - depth
		stand_pat_ready = not board.is_check() 
		curr_val = self.board_valuation(board, player) if player else -self.board_valuation(board, player)
		prev_moves, best_moves = [move], [move]

		if stand_pat_ready and curr_val >= beta:
			return curr_val, prev_moves

		best_val = curr_val if not board.is_check() else -float('inf')
		alpha = max(alpha, curr_val) if not board.is_check() else alpha
		player_move_ranks = {player: partial(self.rank_move, board, player, board_hash) for player in self.players}
		val = None
		end_square = move.to_square
		responses = self.get_captures(board, prev_capture_val) if not board.is_check() else self.get_moves(board)
		responses.sort(key=player_move_ranks[player], reverse=True)

		if not responses:
			val = self.board_valuation(board, player)
			val = -val if not player else val

			return val, prev_moves
		else:
			for response in responses:
				capture_val = get_square_val(board, response.to_square)
				self.make_move(board, response)
				val, next_moves = self.quiescence_search(board, -beta, -alpha, not player, move=response, prev_capture_val=capture_val, depth=depth + 1)
				val = -val
				self.undo_move(board)

				moves =  prev_moves + next_moves
				best_moves = moves if best_val < val else best_moves
				best_val = max(best_val, val)
				alpha = max(alpha, best_val)

				if alpha >= beta:
					height = max_depth - depth 
					self.move_hist[response] += (2 ** (height))
					break

		best_move = best_moves[1] if depth != 0 and len(best_moves) > 1 else (best_moves[0] if best_moves else None)
		has_prev_height = board_hash in self.tt and 'height' in self.tt[board_hash]
		prev_height = self.tt[board_hash]['height'] if has_prev_height else None
		is_best_search = not prev_height or height > prev_height

		if depth == 0 and best_move:
			if not prev_height or height > prev_height:
				self.tt[board_hash]['best_move'] = best_move

		return best_val, best_moves

	def get_quiescence_squares(self, board, move, opponent):
		"""
		Get a list of unstable squares whose next moves should be monitored closely during quiescence search. 
		
		Arguments:
		board (chess Board): Current chess position. 
		move (chess Move): Last move made by opponent.
		opponent (string): Name of opponent.

		Returns:
		A set of chess squares for the quiescence search. 
		"""

		move_hash = str(board) + str(move) + str(opponent)

		if move_hash in self.quiescence_squares_cache:
			return self.quiescence_squares_cache[move_hash]

		opponent_end = move.to_square
		opponent_attacked = get_attacks_on(board, opponent, opponent_end)
		opponent_attacks = get_attacks_from(board, opponent, opponent_end)
		quiescence_squares = set(opponent_attacked + opponent_attacks)
		self.quiescence_squares_cache[move_hash] = quiescence_squares

		return quiescence_squares

	def find_best_checkmate(self, move_stats, winning=True):
		"""
		Arguments:
		move_stats (dict): A dictionary mapping each move to a 2 item list containing 
		the move's value from the latest (deepest) alpha beta search, and the best
		moves following that move that led to the valuation. 
		winning (bool): Whether the AI has a checkmate (True) or its opponent does.

		Returns:
		The optimal move given that one of the players has checkmate. If the AI has
		checkmate, this corresponds to the move that requires the least amount of
		following moves to acheive checkmate. On the other hand, if the opponent 
		has checkmate, the best move is the one that delays checkmate as much as
		possible (most amount of following moves before checkmate).
		"""

		mate_val = float('inf') if winning else -float('inf')
		mate_moves = [move for move in move_stats if move_stats[move][0] == mate_val]
		mate_stats = {move: move_stats[move] for move in mate_moves}
		best_checkmate = sorted(mate_stats.items(), key=lambda x:len(x[1][1]),
							    reverse=not winning)[0]

		return best_checkmate

	def alpha_beta_search(self, board, alpha, beta, player, depth, max_depth, move=None, prev_move=None):
		"""
		Algorithm for finding optimal moves in a two person, zero-sum game. It's
		a variant of the minimax algorithm for minimizing worst case loss in zero
		sum games. What distinguishes alpha beta from other minimax algorithms is
		mainly its alpha and beta variables--which are used to prune non-minimax
		branches before they're analyzed. Alpha represents a lower bound of the
		maximizing player's maximal move value. Beta represents an upper bound
		of the minimizing player's minimal move value.

		In this case, the maximizing and minimizing players are White and Black,
		respectively. Cut-offs occur when alpha >= beta, which indicates that a 
		previously explored opponent move is better than the move currently being
		explored. So there's no point in further exploring branches from this
		move, because this position won't be reached if the players play optimally.

		Arguments:
		board (chess Board): The position to be analyzed.
		alpha (float): Current lower bound of the highest value White can attain.
		beta (float): Current upper bound of the lowest value Black can attain
		(low = better for Black).
		player (string): Active player.
		depth (int): Depth of the current position being explored (how many half
		moves in the future from original position).
		max_depth (int): Max depth of search at which position is evaluated
		directly or with a quiescence search.
		move (chess Move): The most recently made move.

		Returns:
		best_val (float): A float representing an estimate of how favorable the
		best move for the given player is, with respect to White. 
		best_moves (chess Move list): Sequence of best minimax moves (first move
		is Black, second is White, ...) available for player.
		"""

		board_hash = chess.polyglot.zobrist_hash(board)
		player_move_ranks = {player: partial(self.rank_move, board, player,
									 board_hash) for player in self.players}
		best_moves = []
		best_val, val = -float('inf'), None
		height = max_depth - depth

		# If maximum depth is reached, then the current current position's value 
		# returned is evaluated and returned, unless a quiescence search is needed.
		# Quiescence search is needed if the position is unstable, such as if it 
		# occurs before or after a capture.
		if depth == max_depth:
			quiescence_squares = self.get_quiescence_squares(board, move, not player) 
			is_check = board.is_check()
			extended_quiescence = prev_move and not quiescence_squares and not is_check
			quiescence_squares = (self.get_quiescence_squares(board, prev_move, player)
								 if extended_quiescence else quiescence_squares)

			if not quiescence_squares and not is_check:
				best_val = (-self.board_valuation(board, player) if not player
							 else self.board_valuation(board, player))
				best_moves = [move]
			else:
				best_val, best_moves = self.quiescence_search(board, alpha, beta,
									   player, move)

			return best_moves, best_val

		# Continue searching available moves, tracking the best move-value pair
		# for the moving player and breaking the search if alpha >= beta -> not
		# a minimax branch.
		else:
			board_hash = chess.polyglot.zobrist_hash(board)
			board_seen = board_hash in self.tt and 'best_moves' in self.tt[board_hash]
			responses = self.tt[board_hash]['best_moves'] if board_seen else self.get_moves(board)

			if depth <= self.max_sorting_depth and max_depth > 1:
				responses.sort(key=player_move_ranks[player], reverse=True)

			if not responses:
				best_val = self.board_valuation(board, player)
				best_val = -best_val if not player else best_val
				best_moves = [move]

			# See below comment for explanation. 
			if depth == 0:
				response_stats = {}

			for response_ind, response in enumerate(responses):
				self.make_move(board, response)
				val_moves, val = self.alpha_beta_search(board, -beta, -alpha,
								 not player, depth + 1, max_depth, response, move)
				val = -val 
				self.undo_move(board)

				val_moves = [move] + val_moves if depth != 0 else val_moves
				best_moves = val_moves if best_val < val else best_moves
				best_val = max(best_val, val)
				alpha = max(alpha, best_val)

				if depth == 0:
					response_stats[response] = [val, val_moves]

				if alpha >= beta:
					self.move_hist[response] += (2 ** (height))
					break

			# Since ordering initial moves can result in huge alpha beta speedups,
			# the alpha beta search results are stored for each initial move 
			# at each IDS depth, which are then used to sort the initial moves 
			# at the start of the next IDS iteration. 
			if depth == 0:
				response_stats = sorted(response_stats.items(),
									    key=lambda x:x[1][0], reverse=True)
				self.tt[board_hash]['best_moves'] = [x[0] for x in response_stats]

		found_mate = depth == 0 and abs(best_val) == float('inf')

		if found_mate:
			winning = best_val == float('inf')
			best_move = self.find_best_checkmate(dict(response_stats), winning)
		else:
			has_prev_height = board_hash in self.tt and 'height' in self.tt[board_hash]
			prev_height = self.tt[board_hash]['height'] if has_prev_height else None
			best_move = (best_moves[1] if depth != 0 and len(best_moves) > 1
						 else (best_moves[0] if best_moves else None))
			is_best_search = not prev_height or height > prev_height

			if responses and best_move and is_best_search:
				self.tt[board_hash]['best_move'] = best_move
				self.tt[board_hash]['height'] = height

		return best_moves, best_val

	def iterative_depth_search(self, board, player, t_max=30, min_depth=4, stop_at_depth=False):
		"""
		Iteratively find best moves with an alpha beta search, incrementing the
		search depth with each iteration. This is done until a minimum depth and
		elapsed time have been reached. Information from previous searches is
		used to greatly reduce the runtime and search space of subsequent searches.

		Arguments:
		board (chess Board): Position to be analyzed.
		t_max (float): Max time for the last alpha beta search.
		min_depth (int): The minmum depth that must be reached during alpha beta
		searches. Will override t_max if not reached.
		stop_at_depth (bool): Whether the search should stop at min_depth
		regardless of elapsed time.

		Returns:
		best_val (float): A float representing an estimate of how favorable the
		player's best move is with respect to White. 
		best_moves (chess Move list): Sequence of best minimax moves (first move
		is Black, second is White, ...) available for player.
		"""

		t_elapsed = 0.0
		best_move, max_depth = None, 1
		alpha, beta = -float('inf'), float('inf')

		while max_depth <= min_depth or t_elapsed <= t_max:
			if stop_at_depth and max_depth > min_depth:
				break

			start = time.time()
			best_moves, best_val = self.alpha_beta_search(board, alpha, beta, player, 0, max_depth)
			t_elapsed += time.time() - start
			max_depth += 1
			self.update()

			# Checkmate found.
			if abs(best_val) == float('inf'):
				self.moves_til_checkmate = len(best_moves)
				break

		best_move = best_moves[0]

		return best_move, best_val

	def save_valuation_cache(self):
		cache_file = open(self.valuation_path, 'wb')
		cache_file.write(pickle.dumps(self.valuation_cache))
		cache_file.close()

	def scale_move_hist(self):
		"""
		Scale down move history values so that older searches from more distant
		positions are less relevant.
		"""

		self.move_hist = defaultdict(int, {move: value/self.move_hist_scale
										   for move, value in self.move_hist.items()})

	def update(self):
		self.save_valuation_cache()
		self.scale_move_hist()
		move_hist_vals = list(self.move_hist.values())
		self.move_hist_norm = (np.percentile(move_hist_vals, 90) if move_hist_vals
								 else 1.0e-5)
		self.move_hist_max = np.max(move_hist_vals) if move_hist_vals else 1.0e-5

	def get_rel_feature_importance(self, feature_labels=None):
		"""
		Returns a sorted dictionary of feature labels and their relative
		importances with respect to lowest nonzero importance.
		"""

		feature_importances = {i: self.model.feature_importances_[i]
							 for i in range(len(self.model.feature_importances_))}
		feature_importances = dict(sorted(feature_importances.items(),
								   key=operator.itemgetter(1), reverse=True))
		min_importance = min([x for x in self.model.feature_importances_ if x != 0])
		num_features = len(feature_importances)

		if feature_labels:
			labels = get_feature_labels()
			feature_inds = range(len(labels))
			rel_feature_importance = {labels[i]: feature_importances[i] / min_importance
									  for i in feature_inds}
		else:
			rel_feature_importance = {feature: feature_importances[feature] / min_importance
									  for feature in feature_importances}

		return rel_feature_importance

	def compare_logistic_features(self):
		"""
		Returns the relative feature importances of a logistic model's features 
		with respect to the minimal nonzero feature importance. 
		"""

		logistic_coefs = self.model.coef_[0]
		feature_importances = np.exp(logistic_coefs)
		min_feature_importance = np.min(feature_importances[feature_importances > 0])
		relative_importances = feature_importances / min_feature_importance

		return reative_importances


