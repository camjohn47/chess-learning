# chess-explorer
Chess-explorer is about exploring chess with data. You can conveniently use the *ChessPipeline* module to parse through and analyze an arbitrarily large amount of chess games. More specifically, chess game data formatted in PGN files. Careful pipeline design and batch learning methods solve the memory issues which would otherwise certainly be encountered when either processing or modeling hundreds of thousands (or more) of chess games. There is also a *ChessAI* module which determines optimal moves for a given chess position given an evaluation function of your choice. The evaluation function can be self-designed using heuristics or directly loaded from a ML model trained using *ChessPipeline* (or trained somewhere else). 

Collectively, the code is capable of training supervised ML models from millions of expert-level chess games, finding optimal moves for a chess game with an AI engine, and easily playing against the computer with an intuitive interface. This allows for chess learning experimentation through pipeline-based learning and playing against the chess game/AI engines.  More detailed instructions about installation and different features of the project are shown below.

## Table of Contents 
- [Installation](#installation)
- [Features](#features)
- [Support](#support)

## Installation 

### Clone
You can clone this repository by using the following link: https://github.com/camjohn47/chess-data. 

### Prerequisites and Setup
You'll need the following Python modules in order to run all three modules: 
* numpy
* sklearn
* random
* glob
* chess
* math
* pickle
* operator

Once all of these modules are properly installed, you can run any of the modules from within their directory. 

## Features
### Chess Pipeline
A pipeline for parsing and analyzing large amounts of pgn chess files. Pgn is a format in which chess games can be represented, and a pgn file is a file containing multiple pgn chess games. The pipeline is designed with the capability to build certain features from a chess position. These include the amount of each player's piece types on the board, where each piece is located, bishop pairs, pawn development, etc.. However, you can easily write methods to build additional features, insofar as these features can be built with *python-chess*: https://python-chess.readthedocs.io/en/latest/. The module is extensive, so most features should be within *python-chess*'s capabilities. 

Note that in order to learn even the most basic insights about chess from game play, tens of thousands of expert games are very likely a bare minimum. The example training data provided in the directory *training_data* contains hundreds of thousands of expert games, thanks to pgnmentor.com and TWIC: https://theweekinchess.com/twic. This data is used to train a SGDClassifier of your choice with a batch learning approach. You can then test these models with *ChessPipeline* using PGN files located in the directory *test_data*.

To get started with a *ChessPipeline* object, you need to initialize it with the parameter *pgn_directory_train*:the directory in which pgn files for training the model are located. If you wish to load an existing SGDClassifier model, then you can set the optional argument *model_path* to the path in which this model is saved. Otherwise, you'll have to provide an additional parameter *model_args*: the desired configurations for the pipeline's *SGDClassifier* model. Different models can be implemented with very trivial modification to the code, such as decision trees. Here is an example of how to start a ChessPipeline from pgn data in *training_data*, run batch learning on 200,000 of its games, and save the resulting model to *final_test.data*. 

Here is an example of how to initialize a ChessPipeline and perform some batch learning. 
```python
from chesspipeline import ChessPipeline

# <pgn_directory_train> is the directory containing pgn files for training the model. 
pgn_directory_train = 'training_data'

# Path in which model will be saved. 
model_path = 'final_test.data'

# The number of partitions into which the pgn files will be split. The number of batches used during batch learning on each partition. 
num_partitions = 1
num_batches = 1000

# Coefficient determining how strong the model's regularization will be weighted. Higher regularization -> higher penalty for model complexity. 
regularization = 1.0e-3

# The portion of each batch to be used for training the model. Conversely, the remaining portion of the batch is used for later testing.  
train_size = 0.85

# Loss function used to train the model.
loss_function = 'log'
model_args = {'loss':loss_function,'alpha':regularization}

pipeline = ChessPipeline(pgn_directory_train,model_args)
validation_error = pipeline.batch_validation(num_partitions,num_batches,train_size=train_size,model_path=model_path)
```
Since holding the features for hundreds of thousands of games all at once will likely cause memory overflow, the pipeline first randomly partitions the pgn files, after which it processes each partition into chess positions. Then, each partition is transformed into input/output batches for training and testing. Motivation is to ensure that only a small amount of processed batch data is needed at a single time. This way, we can prevent the memory issues that would likely occur if we were to instead build positions and features for the entire collection of pgn data (for all games) at once. 

### Chess AI
AI engine used for calculating opponent moves in the *ChessGame* module. It uses an alpha-beta pruning search to determine optimal moves given a valuation function. The valuation function assesses how good a chess position is for either player. You can choose to use either an ML valuation function or a heuristic based valuation function. Heuristic valuation can be customized by changing the *get_features* method. Input features used for heuristic/model-driven valuation can be customized through modifying *get_heuristic_features/get_model_features*. 

For example, suppose you want the AI valuation to simply be a function of each piece's quantity on the board. Using the chess module, the following *count_pieces* method gets the job done simply. 

  ``` python
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
  ```
  We can test this function by applying it to count the pieces on a starting chess board. 
  ``` python
  import chess
  
  board = chess.Board()
  print(board)
  piece_counts = count_pieces(board)
  pieces = ['P', 'N', 'B', 'R', 'Q','K','p', 'n', 'b', 'r', 'q','k']
  piece_counts = dict(list(zip(pieces,piece_counts)))
  
  print(+ '\n' + 'Piece counts')
  print(piece_counts)    
  ```
  ```
  r n b q k b n r
  p p p p p p p p
  . . . . . . . .
  . . . . . . . .
  . . . . . . . .
  . . . . . . . .
  P P P P P P P P
  R N B Q K B N R

  Piece counts
  {'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1, 'K': 1, 'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1, 'k': 1}
  ```

### Chess Game 
A Python interface for directly playing chess in Terminal. When playing chess games with the *chess_game* script, opponent game play comes from an instance of *ChessAI*. You can use this to test ML models built from a chess pipeline; test any arbitrary chess evaluation function (for example, using heuristics or a ML model trained with *ChessPipeline*); or just play for fun.

Note that this script uses a position cache to store chess positions and their valuations as they're calculated. This means that when changing models, the path should be changed as well. Without doing so, different chess positions can be calculated with different valuation functions in the same game. The following excerpt is taken from the end of the *chess_game* script. This is the code in which changes should be made to reflect new models.

``` python
from chessai import ChessAI

cache_path = 'example_cache.data'
model_path = 'example_model.data'
ai = ChessAI(cache_path=cache_path,model_path=model_path)
```
Here is an example of the game's interface. You'll see the following output from Terminal after successfully launching a game with *python3 chess_game.py*. 
``` 
     Board                         Notation                       Your Available Moves
r n b q k b n r            a8 b8 c8 d8 e8 f8 g8 h8            a2a3, a2a4, b1a3, b1c3, b2b3
p p p p p p p p            a7 b7 c7 d7 e7 f7 g7 h7            b2b4, c2c3, c2c4, d2d3, d2d4
- - - - - - - -            a6 b6 c6 d6 e6 f6 g6 h6            e2e3, e2e4, f2f3, f2f4, g1f3
- - - - - - - -            a5 b5 c5 d5 e5 f5 g5 h5            g1h3, g2g3, g2g4, h2h3, h2h4
- - - - - - - -            a4 b4 c4 d4 e4 f4 g4 h4
- - - - - - - -            a3 b3 c3 d3 e3 f3 g3 h3
P P P P P P P P            a2 b2 c2 d2 e2 f2 g2 h2
R N B Q K B N R            a1 b1 c1 d1 e1 f1 g1 h1
```

It's your turn! Please submit one of the legal moves shown above.

## Support
If you have any questions or feedback, please email curiouscalvinj@gmail.com. Thanks for checking the project out, and thanks to https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46#example-optional for providing a helpful readme template.
