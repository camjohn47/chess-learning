import math 
import numpy as np
import chess
import chess.pgn
from collections import defaultdict
import random
import time
import scipy.sparse.linalg

# Initializing string arrays representing the complete set of chess pieces, as well as subarrays associated with each player's pieces. 	
# Each such string array has a corresponding string-int dictionary whose keys are piece labels that have respective int values equal to the string's index in the given piece string array. 
pieces = ['P', 'N', 'B', 'R', 'Q', 'K','p', 'n', 'b', 'r', 'q', 'k']
white_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
black_pieces = ['p', 'n', 'b', 'r', 'q', 'k']
piece_indices = defaultdict(int)
chess.WHITE = True
i = 0
for piece in pieces:
	piece_indices[piece] = i
	i = i + 1
white_indices = defaultdict(int)
i = 0
for piece in white_pieces:
	white_indices[piece] = i
	i = i + 1	
black_indices = defaultdict(int)
i = 6
for piece in black_pieces:
	black_indices[piece] = i
	i = i + 1
# The center squares will be used later in calculating featurues of the predictive model. Specifically, control of the center: centrality. 
center_squares = [27, 28, 35, 36]

# Declaration of functions. Each function is described individually below. 

# The function 'piece_squares' takes an instance of the chess board class and returns a 12 x 64 int array 'matrix'. 
# Note that the board class, a part of Python's chess package, is the fundamental chess data structure used in this program. It 
# contains all information about the status of a game at a particular turn. As for the 12 x 64 matrix, each row designates
# the positional data for one of the 12 piece types (6 types of pieces for both 2 players). So each row has an array of 64 ints, 
# where for j=1,...,64, the jth index of the ith piece type is a 1 if square j is occupied by piece type i, and 0 otherwise. 
def piece_squares(board):
	matrix = [[]]
	for i in range(0,6):
		white_squares = board.pieces(i + 1, True)
		for square in white_squares:
			matrix.append([i,square])

	for i in range(0,6):
		black_squares = board.pieces(i + 1, False)
		for square in black_squares:
			matrix.append([i + 6,square])
	matrix.remove([])
	return matrix

# Chebyshev distance is a metric which can be imposed on certain discrete topological structures. For vectors u and v, it is given by the maximum 
# distance between u and v with respect to a fixed component. Intuitively, this is useful in the context of chess because it provides a conservative notion of 
# how long it might take slower pieces (pawns, knights, kings) to move between squares. An equivalent definition is the maximum number of king moves required to get from u to v. 
def chebyshev(u, v):
	u_size = len(u)
	v_size = len(v)
	if (u_size != v_size):
		print("Error: Chebyshev vectors have incompatible dimensions.")
	else: 
		diff_max = 0
		for i in range(u_size):
			diff = np.abs(u[i] - v[i])
			if (diff > diff_max):
				diff_max = diff 
	return diff_max

def get_xy(index):
	x = index % 8
	y = index / 8
	return [x,y]

center_vecs = [get_xy(index) for index in center_squares]

# Extraction analyzes a board instance and returns a vector of extra input features which are to be incorporated into the model. 
# Currently, these features are the following: 1) Two (6 x 1) vectors 'white_attacks' and 'black_attacks' whose ith entries are the amounts of
# white and black pieces, respectively, which are attacking opposite player's ith piece. 2) White and black mobility, which indicate the quantity
# of moves available to the given player. 3) The center parameters dictate how important chebyshev distance between the knights and the center of the board 
# is in the prediction model. 4) Finally, 'white_pair' and 'black_pair' are in {0,1}, where a 1 indicates that player has the bishop pair, and a zero indicates 
# that they do not. These features are calculated, and then concatenated into the return vector 'input_vec'.
def extraction(board):
	# Initialization of parameters. 
	white_count = 0
	white_center = 0
	white_bishops = 0
	white_pair = 0
	black_attacks = np.zeros(6)
	# Data for each of white's 6 piece types are analyzed. 
	for i in range(0,6):
		white_squares = board.pieces(i + 1, True)
		for square in white_squares:
			index = (64.0 * i) + square
			black_attacks[i] = len(board.attackers(False, square))
			if (i == 1):
				index_vec = get_xy(index)
				white_center += min([chebyshev(index_vec, center_vecs[i]) for i in range(4)])
		if (i == 2):
				if (len(white_squares) == 2):
					white_pair = 1
	black_count = 0
	black_center = 0
	black_pair = 0
	white_attacks = np.zeros(6)
	# Data for each of black's 6 piece types are analyzed. 
	for i in range(0,6):
		black_squares = board.pieces(i + 1, False)
		for square in black_squares:
			index = (64.0 * (i + 6)) + square
			white_attacks[i] = (len(board.attackers(True, square)))
			if (i == 1):
				index_vec = get_xy(index)
				black_center += min([chebyshev(index_vec, center_vecs[i]) for i in range(4)])
		if (i == 2):
			if (len(black_squares) == 2):
				black_pair = 1
	null = chess.Move.null()
	# Mobilities are calculated by the number of moves each player has--an important aspect of the game. 
	if (board.turn == True):
		white_mobility = board.legal_moves.count()
		board.push(null)
		black_mobility = board.legal_moves.count()
		board.pop()
	else:
		black_mobility = board.legal_moves.count()
		board.push(null)
		white_mobility = board.legal_moves.count()
		board.pop()

	input_extra = np.zeros(18)
	input_extra[0:6] = white_attacks
	input_extra[6:12] = black_attacks 
	input_extra[12] = white_mobility
	input_extra[13] = black_mobility
	input_extra[14] = white_center
	input_extra[15] = black_center
	input_extra[16] = white_pair
	input_extra[17] = black_pair
	return input_extra

# This is the logistic function, which is the fundamental expression of logistic regression. It takes an input vector and parameter, 
# and maps them to a real number in the interval [0,1]. That output is the estimated probability of white winning. 
def logistic(input_vec, parameter):
	dot_prod = np.dot(input_vec, parameter)
	return 1.0 / (1.0 + math.exp(- dot_prod))

# logistic_grad: Gradient of the logistic function above. 
def logistic_grad(inputs, params, output):
	size = inputs.size
	grad = np.zeros(size)
	for i in range(size):
		grad[i] = inputs[i] * (logistic(inputs,params) - output)
	return grad	

# Below is an implementation of newton's method--a common approach to optimizing vector functions 
# when direct computations or closed form solutions are unfeasible. 'inputs' and 'outputs' still refer to input data
# the chess game and the result of that game. 'parameter' is the unknown vector in our predictive model that we're trying 
# to learn. We want the parameter which minimizes the cross entropy between our logistic model and the output distribution. 

def newton_method(inputs, results, parameter, convergence, feature_size, train_size, cut_off):
	start = time.time()
	k = 0
	#param_k: Parameter vector as updated at each iteration k. 
	param_k = parameter
	#grad_k: Gradient of the cost funtion we're trying to minimize, updated at each iteration k. 
	grad_k = np.ones(feature_size)
	# grad_norm: The norm (length) of 'grad_k'. 
	grad_norm = 0
	# result_vec: Vector containing the game results of every game in the training data. 
	result_vec = [results[i] for i in range(train_size)]
	# Main while-loop in which newton's optimization method is implemented. Termination occurs when convergence has reached, or the maximum iteration 'cut_off'.
	while(((grad_norm > convergence) or k == 0) and k < cut_off):
		logistic_vec = [logistic(inputs[i], param_k) for i in range(train_size)]
		logistic_mat = np.diag(logistic_vec)
		inputs_tran = np.transpose(inputs)
		grad_k = np.dot(inputs_tran, np.subtract(logistic_vec, result_vec)) 
		# hessian_k: Hessian matrix of the cost function at the current iteration k. 
		hessian_k = np.matmul(inputs_tran, np.matmul(logistic_mat, inputs))
		# step: Optimal step in which to update the parameter towards optimal parameter, as suggested by the newton's method. 
		step = scipy.sparse.linalg.cg(hessian_k, np.transpose(-grad_k))[0]
		param_k += step
		k += 1
		grad_norm = np.linalg.norm(grad_k)

	return parameter

# Reading training data from PGN file 'pgn' is executed in the for-loop below. For this part of the code, the training data comprised of board states and 
# their results (0: black won, 1: white won, 1/2: draw) are extracted from the PGN chess game data. 
pgn = open('/Users/PickleTickler/Documents/PythonChess/AILocate/Final/TrainingData.pgn')
# inputs: The list which will contain input features for every turn of every game. 
inputs  = []
# Dictionaries are designed so that games and their results can be elegantly and quickly found from game indices.
# For each game, every turn has a respective board and result. 

results = defaultdict(float)
game_dic = defaultdict(int)
game_index = 0
board_index = 0
#'cut_off' is the maximum amount of games to be used in the training process. Optional, but if desired then all code involving this term must be uncommented.
#cut_off = 1000
#The following for loop uses Python's chess package to read and interpret a list containing thousands of professional chess games.
for offset, header in chess.pgn.scan_headers(pgn):
	if(game_index % 5000 == 0 and game_index != 0):
		print("Game index: " + str(game_index))
	#if(game_index == cut_off):
	#	print("Reached cut off.")
	#	break
	game = chess.pgn.read_game(pgn)
	if (header["Result"][1] == '/'):
		result = 0.5
	else:
		result = float(header["Result"][0])
	results[game_index] = result
	board = game.board()
	# Each move is analyzed for each game found in the training data. 
	for move in game.main_line():
		#piece_tables: A 12 x 64 {0, 1} (0 means unoccupied, 1 means occupied) matrix, with 12 rows for each piece type and 64 columns for each square. 
		piece_tables = np.zeros((12,64))
		board.push(move)
		#active_squares: A list of the active squares for each piece type.
		active_squares = piece_squares(board)
		for entry in active_squares:
			index = entry[0]
			square = entry[1]
			piece_tables[index][square] = 1
		# From each board, a 12 x 64 int matrix 'piece_squares' 
		# is flattened into a vector 'input_squares' with 768 components. Once the extra features (additional data traits used in the model) are extracted into 'input_extra', 
		# the entire set of features are concatenated to yield a single, complete input vector 'input_final'. 
		input_squares = piece_tables.flatten()
		input_extra = extraction(board)
		input_final = np.concatenate((input_squares, input_extra))
		inputs.append(input_final)
		game_dic[board_index] = game_index
		board_index += 1
	game_index += 1

# Now the parameters which are to be learned are initialized through 'param_init'. There doesn't appear to be a clear 
# solution for this process. However, an approximate guess can be formed by assigning conventional piece value
# to each of that given piece's associated parameters. For example, the first 64 inputs designate white's 
# pawn piece locations. Common sense says that pawns are worth about 1 point, so setting the first 64 parameters 
# to 1.0 is probably a decent start, although this is certainly not the case. For example, a white pawn on a7 is 
# one square away from queening, and is thus (in most scenarios) worth significantly more than 1 point. As for 
# the extra features, I initialized them based on my estimation of their true value, which I obtained through a 
# combination of trial and error and my own intuition. 

train_size = len(inputs)
print("Number of Games Analyzed: " + str(train_size))
feature_size = len(inputs[0])
print("Number of Features: " + str(feature_size))
param_init = np.zeros(feature_size)
for i in range(384):
	piece = i/64
	if (piece == 0 and i > 7 and i < 56):
		param_init[i] = 1
	elif (piece == 1 or piece == 2):
		param_init[i] = 3
	elif (piece == 3):
		param_init[i] = 5
	elif(piece == 4):
		param_init[i] = 9
	elif(piece == 5): 
		param_init[i] = 100000

for i in range(384, 768):
	piece = i/64
	if (piece == 6 and i > 391 and i < 440):
		param_init[i] = -1
	elif (piece == 7 or piece == 8):
		param_init[i] = -3
	elif (piece == 9):
		param_init[i] = -5
	elif(piece == 10):
		param_init[i] = -9
	elif(piece == 11): 
		param_init[i] = -100000

param_init[768:774] = 0.1
param_init[774:780] = -0.1
param_init[780] = 0.1
param_init[781] = -0.1
param_init[782] = 0.1
param_init[783] = -0.1
param_init[784] = 0.1
param_init[785] = -0.1

# Training the logistic model using the previously constructed chess board input data. As you can see, 
# Newton's method is used. 
# Convergence is somewhat arbitrary, but dictates when the optimization will terminate. More specifically, once the gradient norm is less than the convergence threshold 'convergence', 
# the current parameter is returned as the optimal parameter (parameter which minimzes the cross entropy between the logistic input and 
# output distributions).

# 'step_size': Scalar factor which scales how strongly the parameter changes in response to the local calculations at each iteration.
# 'cut_off': Maximum iteration, at which time newton's method terminates and returns its current estimate of the optimal (minimizing) parameter 'param_op'. 
step_size = 1.0e-3
cut_off = 15
piece_values = defaultdict(float)
convergence = 1.0e-5
param_opt = newton_method(inputs, results, param_init, convergence, feature_size, train_size, cut_off)

# Below, the optimal parameter is written to a .txt file for usage in the AI program. 
f = open('/Users/PickleTickler/Documents/PythonChess/Parameters.txt','w')
for i in range(feature_size):
	f.write(str(param_opt[i]))
	f.write('\n')
f.close()
















