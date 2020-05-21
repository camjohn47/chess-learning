from glob import glob
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import random 
import os
import chess
import chess.pgn
import math
import pickle
import operator

# A pipeline for parsing, analyzing and modeling large amounts of chess data. The data must be expressed in pgn format. 
class ChessPipeline():

	# "pgn_directory" is the directory in which pgn files containing chess game data are located. Pgn is a format used to represent chess games, and a pgn file contains chess games represented in pgn format.
	#  The pipeline possesses a SGDClassifier model, which can be trained to classify chess positions using ML.
	def __init__(self,pgn_directory_train,model_args=None,model_path=None):
		self.pgn_directory_train = pgn_directory_train
		self.pgn_paths_train = glob(os.path.join(pgn_directory_train, "*"))
		self.num_pgns_train = len(self.pgn_paths_train)

		if model_path:
			self.load_model(model_path)

		else:
			self.model = SGDClassifier(**model_args)

		# Five major piece types for each player--excluding kings--where white's pieces are capitalized and black's are in lower-case.
		self.white_pieces = ['P', 'N', 'B', 'R', 'Q']
		self.black_pieces = ['p', 'n', 'b', 'r', 'q']
		self.pieces = self.white_pieces + self.black_pieces
		self.piece_indices = range(1,6)
		self.num_features = len(self.get_features(chess.Board()))

	def save_model(self,model_path):
		model_file = open(model_path,'wb')
		pickle.dump(self.model,model_file)
		model_file.close()

	def load_model(self,model_path):
		model_file = open(model_path,'rb')
		self.model = pickle.load(model_file)
		model_file.close()

	# Determine whether a game's data meets cleanliness criteria. Used in preprocessing to build training data. 
	def headers_filter(self,headers):
		if not headers:
			return False

		elif "Date" not in headers or headers["Result"][0] == '*' or '2' in headers["Result"]:
			return False

		else:
			return True

	# Create a hash key for a game using its headers. Specifically, a game is hashed 
	#using its date and both players. This means that games are uniquely represented 
	#unless they were played on the same date and by the same two players. 
	def hash_game(self,game):
		game_hash = " ".join([game.headers[key] for key in ("Date", "Black", "White")])
		return game_hash

	# Partition a set of pgn paths into <num_partitions> partitions. These partitions 
	# will be equally sized unless the amount of paths is indivisble by <num_partitions>,
	# in which case the last partition will have # of paths = (amount of paths) % <num_paritions>.
	def build_pgn_partitions(self,pgn_paths,num_partitions):
		random.shuffle(pgn_paths)
		num_paths = len(pgn_paths)
		paths_per_partition = int(num_paths/num_partitions)
		partitions = []

		for i in range(num_partitions):
			partition_start,partition_end = i*paths_per_partition,(i+1) * paths_per_partition
			partition = pgn_paths[partition_start:partition_end]
			partitions.append(partition)

		# Edge case which occurs if pgn paths can't be evenly split into <num_partitions> partitions.
		if num_paths % num_partitions != 0:
			partition_start,partition_end = (num_partitions)*paths_per_partition,num_paths
			partition = pgn_paths[partition_start:partition_end]
			partitions.append(partition)

		return partitions

	# Randomly partitions the raw chess position data found in its pgn files. This partitioning is used to prepare mini-batches, which are used for training the classification model.
	# We partition the raw data before building features to prevent memory overflow. Even simple features are too large to be held in memory for > 10,000's of games together (for most computers). 
	def process_pgn_partition(self,pgn_partition,shuffle=True,update_period=100000):
		positions = []
		results = []
		game_index = 0

		for pgn_path in pgn_partition:
			pgn = open(pgn_path,encoding="latin-1")
			print('Processing pgn path ' + str(pgn_path))
			game = chess.pgn.read_game(pgn)

			while game:
				board = game.board()

				# Include the game in processing data only if it satisfies filter criteria. 
				if self.headers_filter(game.headers):
					result = int(float(game.headers["Result"][0]))
					game_hash = self.hash_game(game)

					# Hash the board associated with each position in the game, and assign it and its result to a partition according to this hash. 
					# Note that if the hash were not dependent on the individual moves comprising a game, then a significant degree of randomness would be lost through mapping
					# all positions of a given game to the same partition. The manner in which batch data is organized into batches affects the efficacy and runtime of the batch learning process.
					moves = game.mainline_moves()
					num_moves = len(list(moves))

					if num_moves < 20:
						game = chess.pgn.read_game(pgn)
						continue

					valid_turns = set(range(10,num_moves-10))

					for turn,move in enumerate(moves):
						board.push(move)

						if turn in valid_turns:
							positions.append(board)
							results.append(result) 

					game_index += 1

				if game_index != 0 and game_index % update_period == 0:
					print(str(game_index) + ' games have been prepared for batching.')

				game = chess.pgn.read_game(pgn)

		if shuffle:
			indices = list(range(len(positions)))
			random.shuffle(indices)
			positions = [positions[i] for i in indices]
			results = [results[i] for i in indices]

		return positions,results

	# Count the number of all 12 piece types on the board. There are 6 pieces for
	# each side (white and black): pawn,knight,bishop,rook,queen,king, which are defined 
	# in that order and with white chosen first. 
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

	# Caclculate each player's mobility, where a player's mobility is its number of legal moves. 
	#This feature has incredibly high predictive value of who will win. 
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
		features = piece_counts 

		return features

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

	# Build a set of input features and output features for each game. Output features are simply
	# the result of the game: 1 = white win, 0 = black win. Draws have already been filtered out during 
	# the partioning process. 
	def build_batch(self,boards,batch_outputs,transpose_inputs=False,transpose_outputs=False):
		batch_inputs = []

		for board in boards:
			inputs = self.get_features(board)
			batch_inputs.append(inputs)

		# Randomly shuffle the batch's inputs and outputs. 
		indices = list(range(len(batch_inputs)))
		random.shuffle(indices)
		shuffled_inputs = np.array([batch_inputs[i] for i in indices])
		shuffled_outputs = np.array([batch_outputs[i] for i in indices])

		if transpose_inputs:
			shuffled_inputs = shuffled_inputs.T

		if transpose_outputs:
			shuffled_outputs = shuffled_outputs.T

		return shuffled_inputs,shuffled_outputs

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

	# Validate the ML pipeline by first randomly splitting the training pgn paths into <num_paritions> partitions. 
	# For each partition, first process its pgn files into chess positions and corresponding results. 
	# Then build <num_batches> many batches of inputs/outputs from these positions and results, split each batch into a train/test
	# subset with <train_size>% of the batch data used for training, and the remaining portion of the data stored for later testing.
	# Train the model on each training batch, and finally calculate mean squared error with the fully trained model on all testing batches. 
	def batch_validation(self,num_partitions,num_batches,max_batch_size=1.0e10,train_size=0.85,model_path=None):
		batches = range(num_batches)
		validation_error = 0.0
		test_inputs,test_outputs = np.zeros((0,self.num_features)),np.zeros((0))
		pgn_partitions = self.build_pgn_partitions(self.pgn_paths_train,num_partitions)

		for i,pgn_partition in enumerate(pgn_partitions):
			positions,results = self.process_pgn_partition(pgn_partition,shuffle=True)
			print('Training on partition ' + str(i) + ', paths ' + str(pgn_partition))
			partition_size = float(len(positions))
			batch_size = int(partition_size/num_batches)
			num_partition_wins = float(len([x for x in results if x == 1]))
			num_partition_losses = float(partition_size- num_partition_wins)

			for batch in batches:
				batch_start,batch_end = batch*batch_size,(batch + 1)*batch_size
				batch_inputs,batch_outputs = self.build_batch(positions[batch_start:batch_end],results[batch_start:batch_end])
				batch_training_inputs,batch_test_inputs,batch_training_outputs,batch_test_outputs = train_test_split(batch_inputs,batch_outputs,train_size=train_size)
				train_batch_size = float(len(batch_training_outputs))
				test_inputs = np.concatenate((test_inputs,batch_test_inputs),axis=0)
				test_outputs = np.concatenate((test_outputs,batch_test_outputs),axis=0)
				num_batch_wins = float(len([x for x in batch_training_outputs if x == 1]))
				num_batch_losses = float(train_batch_size - num_batch_wins)
				sample_weights = [partition_size/(2*num_partition_wins) if x == 1 else partition_size/(2*num_partition_losses) for x in batch_training_outputs]
				self.model.partial_fit(batch_training_inputs,batch_training_outputs,classes=[1,0],sample_weight=sample_weights)

				if model_path:
					self.save_model(model_path)

		predicted_outputs = self.model.predict(test_inputs)
		validation_error = mse(test_outputs,predicted_outputs)

		return validation_error	

	# Similar to the above batch validation. The main conceptual difference is that no training occurs, so 
	# each batch of inputs/outputs is not split into train/test subsets. Instead, the entire batch is used for testing. 
	# Returns the average mean squared error across all batches for all partitions of test pgn paths. 
	def test_model(self,pgn_directory_test,num_partitions,num_batches):
		pgn_paths_test = glob(os.path.join(pgn_directory_test, "*"))
		batches = range(num_batches)
		test_inputs,test_outputs = np.zeros((0,self.num_features)),np.zeros((0))
		pgn_partitions = self.build_pgn_partitions(pgn_paths_test,num_partitions)
		total_error = 0.0
		total_batches = 0.0

		for i,pgn_partition in enumerate(pgn_partitions):
			positions,results = self.process_pgn_partition(pgn_partition,shuffle=True)
			print('Test partition ' + str(i) + ', paths ' + str(pgn_partition))
			partition_size = len(positions)
			batch_size = int(partition_size/num_batches)
			print(len(positions))
			num_wins = float(len([x for x in results if x == 1]))
			num_losses = float(partition_size- num_wins)

			for batch in batches:
				batch_start,batch_end = batch*batch_size,(batch + 1)*batch_size
				batch_inputs,batch_outputs = self.build_batch(positions[batch_start:batch_end],results[batch_start:batch_end])
				batch_inputs = np.matrix(batch_inputs)
				predicted_outputs = self.model.predict(batch_inputs)
				batch_error = mse(batch_outputs,predicted_outputs)

				total_error += batch_error
				total_batches += 1.0

		test_error = total_error/total_batches

		return test_error

	# Return a sorted dictionary mapping piece type ---> piece importance, where piece importance
	# is determined by the model coefficients. Note that this must be done after training the model
	# with batch learning, or some arbitrary fitting method (fit or partial_fit).
	def get_piece_importance(self):
		piece_to_importance = {}
		piece_coefficients = self.model.coef_[0]
		print('Model coefficients: ' + str(piece_coefficients))
		white_pawn_importance = np.exp(piece_coefficients[0])
		num_pieces = 10

		for i in range(num_pieces):
			piece_importance = np.exp(piece_coefficients[i])
			relative_importance = piece_importance/white_pawn_importance
			piece = self.pieces[i]
			piece_to_importance[piece] = relative_importance

		piece_to_importance = dict(sorted(piece_to_importance.items(),key=operator.itemgetter(1),reverse=True))

		return piece_to_importance
