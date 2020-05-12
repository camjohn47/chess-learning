import os
import pickle
import chess
import chess.polyglot
import numpy as np
import operator

class ChessAI():
	
	def __init__(self,cache_path,model_path=None):
		# Load position cache from designated path. This is a hash-table lookup from which previously analyzed positions can be retreived.  
		self.position_cache = {}
		self.cache_path = cache_path 

		if os.path.exists(cache_path):
			file = open(cache_path,'rb')
			self.position_cache = pickle.load(file)
			file.close()

		self.piece_indices = range(1,7)
		self.null = chess.Move.null()
		self.piece_values = {1:1,2:3,3:3.3,4:4.2,5:9,6:15}
		white_piece_values = [1,3,3.3,4.2,9,15]
		black_piece_values = [-0.97*x for x in white_piece_values]
		self.piece_values = white_piece_values + black_piece_values
		self.mobility_weight = 0.1
		self.pawn_development_weight = 0.05

		# If model_path is provided, the Ml model serialized in it will be used to evaluate chess positions. Otherwise, a heuristic function will be used. 
		if model_path:
			file = open(model_path,'rb')
			self.model = pickle.load(file)
			file.close()

	# Count the number of all 12 piece types on the board. There are 6 pieces for each side (white and black): pawn,knight,bishop,rook,queen,king, which are defined in that order and with white chosen first. 
	def get_piece_counts(self,board):
		piece_counts = []
		for piece_index in self.piece_indices:
			piece_squares = board.pieces(piece_index, True)
			piece_count = len(piece_squares)
			piece_counts.append(piece_count)

		for piece_index in self.piece_indices:
			piece_squares = board.pieces(piece_index, False)
			piece_count = len(piece_squares)
			piece_counts.append(piece_count)

		return piece_counts

	# Calculate a vector containing white's move count and black's move count.
	def get_mobility(self,board): 
		white_mobility,black_mobility = 0,0

		if board.turn:
			white_mobility = board.legal_moves.count()
			board.push(self.null)
			black_mobility = board.legal_moves.count()
			board.pop()

		else:
			black_mobility = board.legal_moves.count()
			board.push(self.null)
			white_mobility = board.legal_moves.count()
			board.pop()

		return [white_mobility,black_mobility]

	# Pawn development is calculated for each player as the total amount of rows thay player's pawns have traveled. 
	def get_pawn_development(self,board):
		white_pawn_squares = board.pieces(1, True)
		white_pawn_development = sum([int(square/8) for square in white_pawn_squares])
		black_pawn_squares = board.pieces(1, False)
		black_pawn_development = sum([int(square/8) for square in black_pawn_squares])
		pawn_development = [white_pawn_development,black_pawn_development]

		return pawn_development

	# Build input features for a chess position represented by a Python chess board. 
	def get_heuristic_features(self,board):
		piece_counts = self.get_piece_counts(board)
		mobility = self.get_mobility(board)
		pawn_development = self.get_pawn_development(board)
		features = [piece_counts,mobility,pawn_development]

		return features

	def get_model_features(self,board):
		piece_counts = self.get_piece_counts(board)
		mobility = self.get_mobility(board)
		features = np.matrix(piece_counts + mobility)

		return features

	# Evaluate position using heuristics, rather than a data-driven model. 
	def heuristic_valuation(self,board):
		valuation = 0.0
		position_hash = chess.polyglot.zobrist_hash(board)

		if position_hash in self.position_cache:
			return self.position_cache[position_hash]

		piece_counts,mobility,pawn_development = self.get_heuristic_features(board)

		for piece_index,piece_count in enumerate(piece_counts):
			valuation += piece_count * self.piece_values[piece_index]

		valuation += (self.mobility_weight * (mobility[0] - mobility[1])) + (self.pawn_development_weight * (pawn_development[0] - pawn_development[1]))
		self.position_cache[position_hash] = valuation

		return valuation

	# Evaluate position using the instance's ML model. 
	def model_valuation(self,board):
		features = (self.get_model_features(board))
		valuation = self.model.predict_proba(features)[0][1]

		return valuation

	def evaluate_move(self,board,move):
		board.push(move)
		valuation = self.heuristic_valuation(board)
		board.pop()

		return valuation

	# Algorithm for finding optimal moves in a two person, zero-sum game. It's identical to the well-known minimax algorithm, with the exception that it is keeps track of two values: alpha and beta. 
	# Alpha represents the greatest value the max player can acheive from other explored paths. Beta represents the lowest known value the minimizing player can acheive from explored
	# paths. This allows for cut-offs to be made when alpha >= beta, because in such cases, the other player is guaranteed to have a better path available, so further exploring the current node 
	# is redundant. For example, suppose you're playing a chess game and looking 4 moves ahead. When analyzing one of the possible branches, you realize that you might win if your opponent plays a
	# foolish blunder. When considering other moves your opponent could make, it's clear that there are much better ones. As soon as you figure this out, you no longer need to explore the blunder path, 
	# because assuming your opponent plays well, he/she won't do so. You can find more info online. 
	def alpha_beta_search(self,board,alpha,beta,player,depth):
		if depth == 0:
			value = self.heuristic_valuation(board)
			return value

		elif player == 'White':
			for move in sorted(board.legal_moves,key=lambda move:self.evaluate_move(board,move),reverse=True):
				board.push(move)
				value = self.alpha_beta_search(board,alpha,beta,'Black',depth-1)
				board.pop()

				if value > alpha:
					alpha = value

				if alpha >= beta:
					return alpha

			return alpha

		elif player == 'Black':
			for move in sorted(board.legal_moves,key=lambda move:self.evaluate_move(board,move)):
				board.push(move)
				value = self.alpha_beta_search(board,alpha,beta,'White',depth-1)
				board.pop()

				if value < beta:
					beta = value

				if alpha >= beta:
					return beta

			return beta

	def move_optimization(self,board,alpha,beta,depth):
		opt_move = ''
		min_value = 1.0e10
		for move in board.legal_moves:
			board.push(move)
			value = self.alpha_beta_search(board,alpha,beta,'White',depth-1)
			board.pop()

			if value < min_value:
				min_value = value
				opt_move = move

		self.save_position_cache()
		#print('Valuation: ' + str(min_value))

		return opt_move

	def save_position_cache(self):
		cache_file = open(self.cache_path,'wb')
		cache_file.write(pickle.dumps(self.position_cache))
		cache_file.close()


