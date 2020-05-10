import math
import operator
import string
import chess
import chess.polyglot
import pickle
import sys
from ChessAI import ChessAI

# Constants used in the displayed method to produce UI and visuals.
alphabet = string.ascii_lowercase
moves_per_row = 5
reversed_rows = list(reversed(range(8)))
columns = range(8)
column_labels = '|'.join(['a|','b|','c|','d|','e|','f|','g|','h|'])
headers = '     Board' + '            ' + '             Notation' + '            ' + '           Your Available Moves'

# AI's game tree search parameters. 
beta = float('inf')
alpha = -1*beta
depth = 4

# Method called at every turn to show the game's board, notation and available moves to the user. 
def display(board):
	print('\n' + headers)
	moves = sorted([board.uci(move) for move in board.legal_moves])
	num_move_rows = int(len(moves)/moves_per_row)
	for row in reversed_rows:
		row_start,row_end = 8*row, 8*(row+1)
		row_range = range(row_start,row_end)
		row_pieces =[board.piece_at(row) for row in row_range]
		row_symbols = ' '.join([chess.Piece.symbol(piece) if piece else '-' for piece in row_pieces])
		row_label = str(abs(row) + 1)
		square_labels = []

		for column in columns:
			column_label = alphabet[column]
			square_label = column_label + row_label
			square_labels.append(square_label)
		
		labels = ' '.join(square_labels)
		entry = row_symbols + '            ' + labels
		row = abs(7-row)

		if row <= num_move_rows - 1:
			stop = min(len(moves),(row+1)*moves_per_row)
			move_symbols = [str(move) for move in moves[row*moves_per_row:stop]]
			entry += '            ' + ', '.join(move_symbols)

		print(entry)

# Determines how the game has ended. 
def end_game(board):
	display(board)
	print('')
	if board.is_checkmate():

		if board.result()[0] == '1':
			print('Congratulations! You won.')

		else:
			print('You have lost.')

	else:
		print('Stalemate.')

	sys.exit()

def user_turn(board):
	user_move = input('\n' + 'It\'s your turn! Please submit one of the legal moves shown above.' + '\n')
	user_move = chess.Move.from_uci(user_move)

	while user_move not in board.legal_moves:
		user_move = input('\n' + 'Sorry, but that move is illegal. Please try again with one of the legal moves shown above.' + '\n')
		user_move = chess.Move.from_uci(user_move)
		continue

	board.push(user_move)
	display(board)

def computer_turn(board):
	print('\n' + 'Your move has been successfully made. Thinking about next move...' + '\n')
	black_move = ai.move_optimization(board,alpha,beta,depth)
	board.push(black_move)
	display(board)

# Main script that interacts with the user, executes moves, and organizes the development of the game. 
board = chess.Board()
ai = ChessAI('hash.data')
display(board)

while not board.is_checkmate() and not board.is_stalemate():
	user_turn(board)
	computer_turn(board)

print('')
end_game(board)

