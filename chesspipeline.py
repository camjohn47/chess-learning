from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import random 
import chess
import chess.pgn
import math
import pickle

# A pipeline for parsing, analyzing and modeling large amounts of chess data. The data must be expressed in pgn format. 
class ChessPipeline():

	# "pgn_directory" is the directory in which pgn files containing chess game data are located. Pgn is a format used to represent chess games, and a pgn file contains chess games represented in pgn format.
	#  The pipeline possesses a SGDClassifier model, which can be trained to classify chess positions using ML.
	def __init__(self,pgn_directory,model_args=None):
		self.pgn_directory = pgn_directory
		self.pgn_paths = glob(pgn_directory + '/*')
		self.num_pgns = len(self.pgn_paths)
		self.model = SGDClassifier(**model_args)

		# Five major piece types for each player--excluding kings--where white's pieces are capitalized and black's are in lower-case.
		self.white_pieces = ['P', 'N', 'B', 'R', 'Q']
		self.black_pieces = ['p', 'n', 'b', 'r', 'q']
		self.piece_indices = range(1,6)

	# Determine whether a game's data meets cleanliness criteria. Used in preprocessing to build training data. 
	def headers_filter(self,headers):
		if not headers:
			return False

		elif "Date" not in headers or headers["Result"][0] == '*' or '2' in headers["Result"]:
			return False

		else:
			return True

	# Create a hash key for a game using its headers. Specifically, a game is hashed using its date and both players. This means that games are uniquely represented unless 
	# they were played on the same date and by the same two players. 
	def hash_game(self,game):
		hash_features = []
		headers = game.headers
		hash_features.append(headers['Date'])
		hash_features.append(headers['Black'])
		hash_features.append(headers['White'])
		game_hash = ' '.join(hash_features)

		return game_hash

	# Randomly partitions the raw chess position data found in its pgn files. This partitioning is used to prepare mini-batches, which are used for training the classification model.
	# We partition the raw data before building features to prevent memory overflow. Even simple features are too large to be held in memory for > 10,000's of games together (for most computers). 
	def partition_pgn_data(self,num_partitions,downsample,update_period=10000):
		partitions = [ [] for partition in range(num_partitions)]
		game_index = 0
		for pgn_path in self.pgn_paths:

			if game_index > downsample:
				break

			pgn = open(pgn_path,encoding="latin-1")
			game = chess.pgn.read_game(pgn)

			while game:

				if self.headers_filter(game.headers):
					game_hash = self.hash_game(game)
					partition = hash(game_hash) % num_partitions
					partitions[partition].append(game)
					game_index += 1

				if game_index % update_period == 0:
					print(str(game_index) + ' games have been prepared for batching.')

				if game_index > downsample:
					break

				game = chess.pgn.read_game(pgn)

		return partitions

	# Count the number of all 12 piece types on the board. There are 6 pieces for each side (white and black): pawn,knight,bishop,rook,queen,king, which are defined in that order and with white chosen first. 
	def count_pieces(self,board):
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

	# Count bishops for each player to determine whether or not they have the bishop pair. 1 = yes, 0 = no, for each player. 
	def count_bishop_pairs(self,piece_counts):
		white_bishop_pair = 0

		if piece_counts[2] == 2:
			white_bishop_pair = 1

		black_bishop_pair = 0
		
		if piece_counts[8] == 2:
			black_bishop_pair = 1

		return [white_bishop_pair,black_bishop_pair]

	# Build input features for a chess board that will be considered in the model. 
	def get_features(self,board):
		piece_counts = self.count_pieces(board)
		white_mobility,black_mobility = self.get_mobility(board)
		white_bishop_pair,black_bishop_pair = self.count_bishop_pairs(piece_counts)
		postitional_features = [white_mobility,black_mobility]
		features = piece_counts + postitional_features

		return features

	# Caclculate each player's mobility, where a player's mobility is its number of legal moves. This feature has incredibly high predictive value of who will win. 
	def get_mobility(self,board):
		white_mobility,black_mobility = 0,0
		null = chess.Move.null()

		if board.turn:
			white_mobility = board.legal_moves.count()
			board.push(null)
			black_mobility = board.legal_moves.count()
			board.pop()

		else:
			black_mobility = board.legal_moves.count()
			board.push(null)
			white_mobility = board.legal_moves.count()
			board.pop()

		return [white_mobility,black_mobility]

	# For each player's 6 piece types, determine where on the board (=64 squares) all instances of the piece type are located.
	def get_active_squares(self,board):
		active_squares = []
		for piece_index in self.piece_indices:
			white_squares = board.pieces(piece_index + 1, True)
			for square in white_squares:
				active_square = (64*piece_index) + square
				active_squares.append(active_square)

		for piece_index in self.piece_indices:
			black_squares = board.pieces(piece_index + 1, False)
			for square in black_squares:
				active_square = (64*piece_index) + square
				active_squares.append(active_square)

		return active_squares

	# Get features for each position that took place in the game.
	def process_game(self,game):
		inputs = []
		outputs = []
		headers = game.headers
		result = int(float(headers["Result"][0]))
		board = game.board()

		for move in game.mainline_moves():
			board.push(move)
			board_features = self.get_features(board)
			inputs.append(board_features)
			outputs.append(result)

		return [inputs,outputs]

	# Build a set of input features and output features for each game. Output features are simply the result of the game: 1 = white win, 0 = black win. Draws have already been filtered out during 
	# the partioning process. 
	def build_batch(self,games):
		transformed_inputs = []
		transformed_outputs = []

		for game in games: 
			inputs,outputs = self.process_game(game)
			transformed_inputs += inputs
			transformed_outputs += outputs
			inputs,outputs = None,None

		indices = list(range(len(transformed_inputs)))
		random.shuffle(indices)
		shuffled_inputs = [transformed_inputs[i] for i in indices]
		shuffled_outputs = [transformed_outputs[i] for i in indices]

		return [shuffled_inputs,shuffled_outputs]

	# First, partition the raw pgn data into random low-memory game partitions. This is done to avoid memory issues. If we were to isntead try partioning the input features into batches,
	# this would likely result in memoryoverflow issues for 10,000's of games or more, on a typical computer. 
	def batch_learning(self,num_partitions,model_path,downsample=None):
		partitions = self.partition_pgn_data(num_partitions,downsample)

		for i,partition in enumerate(partitions):
			print('Batch ' + str(i))
			batch_inputs,batch_outputs = self.build_batch(partition)
			self.model.partial_fit(np.matrix(batch_inputs),batch_outputs,classes=[1,0])
			self.save_model(model_path)

	def save_model(self,model_path):
		model_file = open(model_path,'wb')
		pickle.dump(self.model,model_file)
		model_file.close()

	# Return a sorted dictionary mapping piece type ---> piece importance, where piece importance is determined by the model coefficients. Note that this must be done after training the model
	# with batch learning, or some arbitrary fitting method (fit or partial_fit).
	def show_piece_importance(self):
		piece_importance = {}
		feature_weights = self.model.coef_[0]

		for i,piece in self.pieces:
			importance = feature_weights[i]
			piece_importance[piece] = importance

		piece_importance = sorted(piece_importance)



		



