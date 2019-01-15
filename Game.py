from __future__ import print_function
import math 
import numpy as np
import chess
import chess.pgn
from collections import defaultdict
import random

#param_data: A list of strings containing the optimal parameter data, read from the parameter text file.
param_data = open('/Users/PickleTickler/Documents/PythonChess/Parameters.txt')
#feature_size: Size of the parameter vector chosen for logistic regression model. 
feature_size = 0
param_dic = defaultdict(float)
# For-loop below reads each entry of the txt file and stores it in the parameter vector. The txt file should
# contain the optimal parameter, which has learned through one of the learning algorithms
for line in param_data:
	entry = line.split()
	param_dic[feature_size] = float(entry[0])
	feature_size += 1

parameter = np.zeros(feature_size)
for j in range(feature_size):
	parameter[j] = param_dic[j]

# Please see notes on any of these functions in either of the learning programs. 
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

def logistic(input_vec, parameter):
	dot_prod = np.dot(input_vec, parameter)
	return 1.0 / (1.0 + math.exp(- dot_prod))

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

center_squares = [27, 28, 35, 36]
center_vecs = [get_xy(index) for index in center_squares]

def extraction(board):
	white_count = 0
	white_center = 0
	white_bishops = 0
	white_pair = 0
	black_attacks = np.zeros(6)
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

def evaluate (board):
	value = 0.0
	piece_tables = np.zeros((12,64))
	active_squares = piece_squares(board)
	for entry in active_squares:
		index = entry[0]
		square = entry[1]
		piece_tables[index][square] = 1

	input_squares = piece_tables.flatten()
	input_extra = extraction(board)
	input_final = np.concatenate((input_squares, input_extra))
	exponent = np.dot(input_final, parameter)
	prob = logistic(input_final, parameter)
	return prob

# Alpha beta is a zero-sum game optimization algorithm whose run time is less than or equal to conventional minimax optimization. 
# It's somewhat complicated, but the basic idea is that alpha beta is identical to minimax, except for its consideration of alpha 
# and beta values, which represent the minimum value the maximizing player can obtain and the maximum value the minimizing player can reach. 
# If alpha >= beta at any point, then one of the players wouldn't allow that particular branch to be chosen, so no point in exploring further. 
def alpha_beta(board, depth, alpha, beta, player):
	if (depth == 0):
		value = evaluate(board)
		return [None, value]
	#Player 1 is the maximizing player: white. 
	elif (player == 1):
		move_values = defaultdict(list)
		#This for-loop iterates over all of white's moves, recursively analyzing and exploring the game tree. 
		for move in board.legal_moves:
			board.push(move)
			value = evaluate(board)
			board.pop()
			if value in move_values:
				move_list = move_values[value]
				move_list.append(move)
				move_values[value] = move_list
			else:
				move_list = []
				move_list.append(move)
				move_values[value] = move_list
		# Sorted alpha beta pruning is almost guaranteed to work better than non-sorted. That is, the move-value dictionary's
		# keys (the moves) are sorted ascendingly by their values. The idea is that the extra time required to evaluate the entire list of moves 
		# prematurely is almost always worth it, because pruning is much more likely to occur when sorting takes place. 
		sorted_values = sorted(move_values.iterkeys())
		sorted_moves = []
		for value in sorted_values:
			moves = []
			moves = move_values[value]
			sorted_moves.append(moves)
		for move_list in sorted_moves:
			for move in move_list:
				board.push(move)
				value = alpha_beta(board, depth - 1, alpha, beta, 0)
				board.pop()
				if value > alpha:
					alpha = value
				if alpha >= beta:
					return [None, beta]
		return [move, value]

	# Player 0 is the minimizing player: black. 
	elif (player == 0):
		move_values = defaultdict(list)
		for move in board.legal_moves:
			board.push(move)
			value = evaluate(board)
			board.pop()
			if value in move_values:
				move_list = move_values[value]
				move_list.append(move)
				move_values[value] = move_list
			else: 
				move_list = []
				move_list.append(move)
				move_values[value] = move_list
		# Black's moves are sorted, but this time, the moves are sorted descendingly by their values. 
		sorted_values = sorted((move_values.iterkeys()), reverse = True)
		sorted_moves = []
		for value in sorted_values:
			moves = []
			moves = move_values[value]
			sorted_moves.append(moves)
		for move_list in sorted_moves:
			for move in move_list:
				board.push(move)
				value = alpha_beta(board, depth - 1, alpha, beta, 1)
				board.pop()
				if value < beta:
					beta = value
				#Termination condition, because alpha>=beta. 
				if alpha >= beta:
					return [None, alpha]
		return [move, value]

#AI is a simple function. It calls on alpha_beta for the optimal move, using a set of specified configurations.
#Since alpha is the min of a supremum (maximizing player's cost function) and beta is the max of an infinimum, it makes
#sense to initialize alpha very low and beta very high. 
def AI(board):
	alpha = -1000000.0
	beta = 1000000.0
	depth = 3
	value = 0.0
	[opt_move, value] = alpha_beta(board, depth, alpha, beta, 0)
	print('move: ' + str(opt_move))
	print('value: ' + str(value))
	return opt_move


## Here we have the main script of the game. At each set of turns, the user is asked to enter a move (in algebraic chess notation). 
## After which, the computer's AI determines an optimal move using alpha beta pruning and its trained predictive model. 
game = chess.pgn.Game()
board = game.board()
print(board)
while not game.board().is_game_over():
	print('Your moves: ' + str(board.legal_moves))
	entry = raw_input('Enter your move in algebraic chess notation. ')
	user_move = str(entry)
	game.add_main_variation(user_move)
	board.push_san(user_move)
	print('\n')
	print(board)
	print('Thinking...')
	AI_move = AI(board)
	print('Computer played: ' + str(AI_move))
	game.add_main_variation(chess.Move.from_uci(str(AI_move)))
	board.push(AI_move)
	print('\n')
	print(board)




	
