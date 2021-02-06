from glob import glob
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.tree import DecisionTreeClassifier as dtc
import sys
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import random 
import os
import chess
import chess.pgn
import pickle
import operator
from chess_features import get_features, get_castling_type
from itertools import product
from functools import partial
from collections import Counter
import chess.polyglot
import time
import math
import shutil

class ChessPipeline():
	"""
	An ML pipeline for analyzing and building predictive features from large amounts of chess games. These features
	can be directly used by ChessPipeline to train classifiers for predicting and evaluating chess positions while 
	avoiding memory overflow and runtime issues. Has been successfully tested on 10's millions of chess games.
	"""

	def __init__(self, model_path, model_type='rfc', model_args={}, reset_model=True):
		"""
		Arguments: 
		model_path (string): Name of file used for saving the ChessPipeline's model. If reset_model=False, then it will
		initialize its model by loading from model_path, rather than starting a new model.
		model_type (string): Acronym used to determine the instance's model category. Can be 'rfc' for random forest classifier, 
							'sgd' for stochastic gradient descent classifier, or 'dtc' for a decision tree classifier.
		model_args (hash): 	 Dictionary mapping a model argument to its desired value.
		reset_model (bool) : Whether the intialized model should be started fresh or loaded from <model_path>.
		"""

		self.model_builders = {'rfc': partial(rfc, warm_start=True, n_estimators=1),
							   'sgd': partial(sgd, warm_start=True),
							   'dtc': dtc}

		if not reset_model:
			self.model = self.load_model(model_path)
		elif model_type in self.model_builders:
			self.model = self.model_builders[model_type](**model_args)
		else:
			print(f'ERROR: ChessPipeline was initialized with invalid model type "{model_type}". The valid model types are "rfc", "sgd", and "dtc".')
			sys.exit(0)

		self.model_path = model_path
		self.model_type = model_type
		self.white_pieces, self.black_pieces = ['P', 'N', 'B', 'R', 'Q', 'K'],  ['p', 'n', 'b', 'r', 'q', 'k']
		self.pieces = self.white_pieces + self.black_pieces
		self.piece_to_index = {self.pieces[i]:i for i in range(len(self.pieces))}
		self.piece_indices = range(1,7)
		self.num_piece_types = len(self.white_pieces)
		self.num_features = len(get_features(chess.Board()))
		self.min_elo = 1800
		self.max_elo = 3000
		self.min_training_move = 3

	def save_model(self):
		"""	Serialize ML model into binary information with pickle, and store it in the instance's model path."""

		model_file = open(self.model_path,'wb')
		pickle.dump(self.model, model_file)
		model_file.close()

	def load_model(self, model_path):
		"""	Return a deserialized ML model from the given model path. NOTE: By default, the model's file must be written in binary mode."""

		model_file = open(model_path,'rb')
		model = pickle.load(model_file)
		model_file.close()

		return model

	def update_model_params(self, params):
		""" Reset the ChessPipeline's random forest classifier to have different hyperparameters. NOTE: This will remove any information the
		 rfc has previously learned.
		
		Arguments:
		rfc_args (hash): A dictionary mapping rfc hyperparameters to their fixed values.
		 """

		self.model = self.model_builders[self.model_type](**params, warm_start=True, n_estimators=1) 

	def headers_filter(self,headers):
		""" Determine whether a game's data meets cleanliness criteria. Used in preprocessing to build training data. 

		Arguments:
		min_elo (int): The minimum ELO score both players must have for the game to be considered.
		max_elo (int): The maximum ELO score both players can have without the game being filtered out.

		Returns:
		bool: Whether the game has passed all filtering criteria.
		"""

		if not headers:
			return False
		elif ("Date" not in headers or headers["Result"][0] == '*' or '2' in
		 	  headers["Result"] or "BlackElo" not in headers or "WhiteElo" not in headers):
			return False
		else:
			try:
				black_elo, white_elo = int(float(headers['BlackElo'])), int(float(headers['WhiteElo'])) 
				white_elo_valid = self.min_elo < white_elo < self.max_elo
				black_elo_valid = self.min_elo < black_elo < self.max_elo

				if black_elo_valid and white_elo_valid:
					return True

			except ValueError:
				return False

	def group_shuffle(self, features, results, max_ind=None):
		""" 
		Shuffle a feature-result batch so that their shuffled indices align. This
		means that the feature result arrays are shuffled together so that they
		maintain their association after shuffling--ie, features_i and results_i
		still correspond to the same position after shuffling, it's just likely a
		different one. 

		Arguments:
		features (np array): float array of numerical features built from chess positions.
		results (np array or list): binary group of results associated with same
		positions in features.
		max_ind (int): the feature/result index past which data is discarded. 
		Used as a cleaning tool to filter out zeros from incomplete batches.

		Returns: 
		features (np array): shuffled features array.
		results (np array/list): shuffled results array, where the shuffling is
		identical to features' shuffling.
		"""

		max_ind = len(features) if not max_ind else max_ind
		inds = list(range(max_ind))
		random.shuffle(inds)

		return features[inds, :], results[inds]

	def serialize_batch(self, features, results, batch_paths):
		"""
		Randomly serialize the individual features/results of the features/results
		batch into the files found in batch_paths.

		Arguments:
		batch_paths ([string]): List of strings associated with paths into which
		features/results should be distributed.
		"""

		features, results = self.group_shuffle(features, results)
		num_batches = float(len(batch_paths))
		features_per_batch = len(features)/num_batches

		for i, batch_path in enumerate(batch_paths):
			if not os.path.exists(batch_path):
				with open(batch_path, 'w') as f:
					pass

			batch_inds = list(range(int(i * features_per_batch), int((i + 1) * features_per_batch)))
			file = open(batch_path, 'ab')
			pickle.dump([features[batch_inds, :], results[batch_inds]], file)
			file.close()

	def get_batch_weights(self, results):
		"""
		Calculates and returns weights from a batch's results.

		Arguments:
		results (np array or list): binary group of results for the batch's positions.

		Returns:
		batch_weights (np array): float array of weights [weight_i], where weight_i = inverse frequency of result_i.
		"""

		result_counts = Counter(results)
		num_results = float(len(results))
		inv_result_prob = {result: num_results/(2 * float(count)) for result, count in result_counts.items()}
		batch_weights = np.array([inv_result_prob[result] for result in results])

		return batch_weights

	def build_batches(self, pgn_dir, batch_dir, num_batch_files, reset_batches=False,
				      batch_size=int(1.0e5), max_batches=int(1.0e3)):
		""" 
		Builds and serializes feature/result batches from the games found in
		<pgn_dir>'s pgn files. These batches can be directly used to train ML
		models with any of ChessPipeline's different learning methods. Batches
		are randomly distributed to different batch files to ensure that learning
		from each batch isn't biased towards certain games. Batches are periodically
		serialized and deleted to avoid memory overflow problems and reduce runtime.

		Arguments:
		pgn_dir (string): A string representing directory of pgn files used for
		building batches.
		batch_dir (string list): Directory in which input/output batches will be saved. 
		num_batch_files (int): Number of batch files that data will be written to. 
		reset_batches (bool): If true, and batches_dir exists, all previous
		batches in batches_dir will be deleted.
		batch_size (int): Maximum size of a features/results batch before it is
		 				  distributed, serialized, and reset.
						  Note: There's a runtime/memory tradeoff in that low 
						  batch sizes need more frequent writing to files, but 
						  large batch sizes require more memory.
		max_batches (int): Maximum number of feature batches written to batch files. 
		"""

		if not os.path.isdir(batch_dir):
			os.mkdir(batch_dir)
		elif reset_batches:
			shutil.rmtree(batch_dir)
			os.mkdir(batch_dir)

		batch_paths = [f"{batch_dir}/batch_{i}.data" for i in range(1, num_batch_files)]
		pgn_paths = glob(os.path.join(pgn_dir, "*"))
		random.shuffle(pgn_paths)
		batch_features = np.zeros((batch_size, self.num_features))
		batch_results = np.zeros(shape=(batch_size))
		curr_batch_size, num_batches, num_positions = 0, 0, 0
		start = time.time()

		for pgn_path in pgn_paths:
			if num_batches > max_batches:
				break

			pgn = open(pgn_path,encoding="latin-1")
			game = chess.pgn.read_game(pgn)

			while game and num_batches <= max_batches:
				board = game.board()
				num_moves = len(list(game.mainline_moves()))

				# If the current batch is too full to ingest data from the next game, it is serialized and deleted.
				# Then a new batch is started.
				if curr_batch_size + num_moves >= batch_size:
					batch_features, batch_results = self.group_shuffle(batch_features,
												    batch_results, curr_batch_size)
					self.serialize_batch(batch_features, batch_results, batch_paths)
					batch_features  = np.zeros((batch_size, self.num_features))
					batch_results = np.zeros(shape=(batch_size))
					num_positions = num_positions + curr_batch_size
					curr_batch_size, num_batches = 0, num_batches + 1
					batch_time, start = time.time() - start, time.time()
					print(f"Number of built positions so far = {num_positions}. Batch build time = {batch_time}.")

				# Filter out games that are incorrectly formatted or ended in a draw.
				if self.headers_filter(game.headers):
					result = int(float(game.headers["Result"][0]))
					moves = game.mainline_moves()
					num_moves = len(list(moves))

					# Ignore empty games.
					if not num_moves:
						game = chess.pgn.read_game(pgn)
						continue

					has_castled = np.zeros((2))

					# Iterate through game's moves and build features/results from each position.
					for move_ind, move in enumerate(moves):
						if board.is_castling(move): 
							player = 0 if board.turn else 1
							castling_type = get_castling_type(move)
							has_castled[player] = castling_type

						board.push(move)

						batch_features[curr_batch_size, :] = get_features(board, has_castled)
						batch_results[curr_batch_size] = result 
						curr_batch_size += 1

				game = chess.pgn.read_game(pgn)

		# Leftover batch to be serialized. 
		if curr_batch_size > 0:
			batch_features, batch_results = self.group_shuffle(batch_features, batch_results, max_ind=curr_batch_size)
			self.serialize_batch(batch_features, batch_results, batch_paths)

	def deserialize_batches(self, batch_dir, batch_size=int(1.0e6), max_batches=int(1.0e3)):
		"""
		Yield batches of desired size from the batch files found in batch_paths.
		Batches are loaded and learned from individually to prevent memory
		issues and for quicker, more effective learning. Batch weights are
		calculated for improved learning (win/loss imbalance reflected in weights).

		Arguments:
		batch_dir (string): Name of directory from which batch data will be loaded. 
		max_batches (int): Maximum number of batches returned (in total) by the
		function call.

		Returns (on each yield iteration):
		batch_features (np float array): Numerical features for random group of
										 chess positions.
		batch_results (binary np array): Results of the same random group of
										 chess positions.
		batch_weights (np float array): Inverse class frequency weights for the
										 same chess positions.
		"""

		batch_paths = glob(os.path.join(batch_dir, "*"))
		random.shuffle(batch_paths)
		start = time.time()
		batch_features = np.zeros((batch_size, self.num_features))
		batch_results = np.zeros(shape=(batch_size))
		curr_batch_size, num_batches = 0, 0

		for j, batch_path in enumerate(batch_paths):
			if num_batches > max_batches:
				break

			file = open(batch_path, 'rb')

			while True and num_batches <= max_batches:
				try:
					item = pickle.load(file)
				except:
					break

				if type(item[0]) == int:
					print(f'Invalid item found with int type at first index: {item}.')
					continue

				features, results = item
				num_item_samples = len(features)

				if curr_batch_size + num_item_samples >= batch_size:
					batch_features, batch_results = self.group_shuffle(batch_features, batch_results, max_ind=curr_batch_size)
					batch_weights = self.get_batch_weights(batch_results) 
					yield batch_features, batch_results, batch_weights

					batch_features, batch_results = np.zeros((batch_size, self.num_features)), np.zeros(shape=(batch_size))
					curr_batch_size, num_batches = 0, num_batches + 1
					del batch_weights

				batch_inds = range(curr_batch_size, curr_batch_size + num_item_samples)
				batch_features[batch_inds, :], batch_results[batch_inds] = features, results
				curr_batch_size += num_item_samples
				del features, results

			file.close()

		batch_features, batch_results = self.group_shuffle(batch_features, batch_results, max_ind=curr_batch_size)
		batch_weights = self.get_batch_weights(batch_results)

		yield batch_features, batch_results, batch_weights

	def unify_batches(self, batch_dir, batch_size=int(1.0e5), max_samples=int(1.0e7)):
		"""
		Loads features/results from the files in batch_dir and collects them into
		a single features/results batch. This method is needed in place of the
		<deserialize_batches> method when training a decision tree classifier with 
		the pipeline, because decision trees are unsuitable for batch learning.
		Since the <tree_learning> method for training dtcs relies on a single 
		fitting call for learning, it can only learn from a single training set.

		Arguments:
		batch_size (int): Same as above, the size of individual feature/result
		batches loaded from batch_paths.
		max_samples (int): The maximum size of the unified batch before it is
		used for learning.

		Returns:
		features, results, weights (np arrays): Unified batch of features,
		results, and inverse class frequency weights.
		"""

		features = np.zeros(shape=(max_samples, self.num_features))
		results, weights = np.zeros((max_samples)), np.zeros((max_samples)) 
		num_samples = 0

		for j, batch in enumerate(self.deserialize_batches(batch_dir, batch_size)):
			batch_features, batch_results, batch_weights = batch
			num_batch_samples = len(batch_features)
			batch_cutoff = (max_samples - num_samples if num_samples + num_batch_samples > max_samples
						   else num_batch_samples)

			batch_inds = range(0, batch_cutoff)
			feature_inds = range(num_samples, min(max_samples, num_samples + num_batch_samples))
			features[feature_inds, :]  = batch_features[batch_inds, :]
			results[feature_inds] = batch_results[batch_inds]
			weights[feature_inds] = batch_weights[batch_inds]
			num_samples += batch_cutoff
			print(f'Batch {j + 1} has been merged. Number of samples = {num_samples}.')

			if num_samples >= max_samples:
				break

		print("Unification of batches has completed.")

		return features, results, weights

	def batch_learning(self, batch_dir, batch_size=int(1.0e5), max_batches=int(1.0e3)):
		"""
		Batch learning approach for incrementally learning from the different
		batches found in train_dir. Note that this is currently only valid for 
		stochastic gradient descent and random forest classifers, which have
		<model_type> = 'sgd' and 'rtc' in ChessPipeline.

		SGD uses batch learning in a straightforward way. Its learning process
		already consists of incremental updates to parameters from individual
		training samples. ChessPipeline implements rfc batch learning by training
		a new tree for each features/results batch, using information from the
		forest to guide the next tree's learning.

		Arguments:
		batch_dir (string): Name of directory from which batch files are loaded.
		batch_size (int): Size of batches used for learning.
		max_batches (int): Max # of feature/result batches that will be
						   used during learning.
		"""

		learning_samples = 0
		batches = self.deserialize_batches(batch_dir, batch_size, max_batches)

		for i, batch in enumerate(batches):
			batch_features, batch_results, batch_weights = batch

			if self.model_type == 'rfc':
				self.model.fit(batch_features, batch_results, sample_weight=batch_weights)
				self.model.n_estimators += 1
			elif self.model_type == 'sgd':
				self.model.partial_fit(batch_features, batch_results,
									   classes=[0,1], sample_weight=batch_weights)
			else:
				print((f"Error: A decision tree classifier can't use batch learning."
					  f" Use <tree_learning> or change the ChessPipeline's model type."))
				break

			learning_samples += len(batch_features)
			self.save_model()
			print((f"Batch {i + 1} learning complete. Number of chess positions "
				  f"used in batch learning so far: {learning_samples}."))

	def tree_learning(self, batch_dir, batch_size=int(1.0e5), max_samples=int(1.0e7)):
		"""
		Train a decision tree classifier with a single fitting call, using the 
		unifed features/results built from the files in <batch_dir>.

		Arguments:
		batch_dir (string): Name of directory from which test inputs/outputs will be built. 
		batch_size (int): Size of individual batches fed into unified batch.
		max_samples (int): Maximum size of the single unified batch used for training the dtc.
		"""

		random.shuffle(batch_paths)
		features, results, weights = self.unify_batches(batch_dir, batch_size, max_samples)
		self.model.fit(new_features, results, sample_weight=weights)
		self.save_model()

	def test_model(self, batch_dir, model=None, batch_size=int(1.0e5), max_batches=10):
		"""
		Returns the average MAE error of a trained model on batches loaded from
		 <batch_dir>.

		Arguments:
		batch_dir (string): Name of directory from which test inputs/outputs will
							be built. 
		model (sklearn model): ML classifier whose predictions on chess positions
							   are tested. Set to pipeline's model by default.
		max_batches (int): Max number of batches used for testing.
	
		Returns:
		avg_error (float): Portion of results that were incorrectly predicted.
		"""

		model = model if model else self.model
		batch_errors, actual_batch_sizes = [], []
		batches = self.deserialize_batches(batch_dir, batch_size, max_batches)

		for i, batch in enumerate(batches):
			if i >= max_batches:
				break

			batch_features, batch_results, _ = batch
			actual_batch_size = len(batch_features)
			actual_batch_sizes.append(actual_batch_size)
			predicted_results = model.predict(batch_features)
			batch_error = mae(batch_results, predicted_results) 
			batch_errors.append(batch_error)

		total_size = np.sum(actual_batch_sizes)
		batch_weights = [actual_batch_size / total_size 
						 for actual_batch_size in actual_batch_sizes]
		avg_error = np.dot(batch_weights, batch_errors)

		return avg_error

	def grid_search(self, train_dir, test_dir, param_grid, batch_sizes,
		  		    const_params={}, model_type=None, max_train_batches=10, max_test_batches=5):
		"""
		A grid search for finding a classifier's optimal hyperparameters
		Conceptually, the same idea as sklearn's grid search. For each 
		combination of hyperparameter values that can be taken from the hyperparameter
		domains in param_grid, a 
		
		Arguments: 
		train_dir (string): Names of directory used for building training batches.
		test_dir (string): Name of directory used for building test batches.
		param_grid (hash): Dictionary of the following form:
		 					{hyperparameter (str): hyperparameter values (list)}. 
		const_params (hash): Dictionary of this form form:
							{hyperparameter label: constant value of hyperparameter}.
		"""

		if not model_type:
			model_type = self.model_type
		elif model_type not in self.model_builders:
			print(f"ERROR: {model_type} is not a valid model type.")
			sys.exit(0)

		param_types = list(param_grid.keys())
		param_combs = list(product(*list(param_grid.values())))
		num_params = len(param_types)
		clf_dics = [{param_types[j]: param_comb[j] for j in range(num_params)}
					 for param_comb in param_combs]

		clfs = [self.model_builders[model_type](**clf_dic, **const_params)
			   for clf_dic in clf_dics]

		clf_errors = []

		if self.model_type == 'dtc':
			for batch_size in batch_sizes:
				features, results, weights = self.unify_batches(train_dir, batch_size)

				for clf_ind, clf in enumerate(clfs):
					clf.fit(features, results, sample_weight=weights)
					error = self.test_model(test_dir, clf, batch_size, max_test_batches)
					clfs[clf_ind] = clf
					clf_dic = clf_dics[clf_ind]
					clf_errors.append([clf_dic, batch_size, error])
					print((f"Hyperparameters = {clf_dic} with batch size ="
						  f" {batch_size} has error = {error}."))

		else:
			for batch_size in batch_sizes:
				batches = self.deserialize_batches(train_dir, batch_size, max_train_batches)
				for i, batch in enumerate(batches):
					batch_features, batch_results, batch_weights = batch

					for clf_ind, clf in enumerate(clfs):
						if self.model_type == 'rfc':
							clf.fit(batch_features, batch_results, sample_weight=batch_weights)
							clf.n_estimators += 1
						else:
							clf.partial_fit(batch_features, batch_results,
										    classes=[0,1], sample_weight=batch_weights)

					print(f'Clfs have been trained on batch {i + 1}.')

				for i, clf in enumerate(clfs):
					error, clf_dic = self.test_model(test_dir, clf, batch_size,
													 max_test_batches), clf_dics[i]
					clf_errors.append([clf_dic, batch_size, error])
					print((f"Parameter combination = {clf_dic} with batch size ="
						  f" {batch_size} has error = {error}."))

			clf_errors = sorted(clf_errors, key=operator.itemgetter(2))

		return clf_errors


