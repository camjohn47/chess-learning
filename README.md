# chess-data
Chess-data is about exploring chess with data. ML-driven pipelines are designed to extract insights from large amounts of chess game data. Collectively, the code can train supervised ML models from hundreds of thousands of chess games, and run chess games using an optimal AI system built from these models. This allows for chess learning models to be tested and experimented with through appropriate modeling and playing. More detailed instructions about installation and different features of the project are shown below.

## Table of Contents (Optional)

> If your `README` has a lot of info, section headers might be nice.

- [Installation](#installation)
- [Features](#features)

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

Once all of these modules are properly installed, you can run any of the modules from within their directory. 

## Features
### Chess Pipeline
A pipeline for parsing and analyzing large amounts of pgn chess files. Pgn is a format in which chess games can be represented, and a pgn file is a file containing multiple pgn chess games. The pipeline is designed with the capability to build certain features from a chess position. These include the amount of each player's piece types on the board, where each piece is located, bishop pairs, pawn development, etc.. However, you can easily write methods to build additional features, insofar as these features can be built with *python-chess*: https://python-chess.readthedocs.io/en/latest/. The module is extensive, so most features should be within *python-chess*'s capabilities. 

Note that in order to find any meaningful insights about chess from game play, tens of thousands of expert games are very likely a bare minimum. The example training data provided in the directory *training_data* contains hundreds of thousands of expert games thanks to pgnmentor.com.

To get started with a *ChessPipeline* object, you need to initialize it with two parameters: *pgn_directory* and *model_args*. The former is a directory in which pgn files are located; the latter consists of the desired ML model configurations. Note that the ML model is a *SGDClassifier*. Different models can be implemented without much modification to the code. Here is an example of how to start a ChessPipeline from pgn data in *training_data*, run batch learning on 200,000 of its games, and save the resulting model to *final_test.data*. 

```python
from chesspipeline import ChessPipeline

# pgn_directory is the directory containing pgn files. 
pgn_directory = 'training_data'
num_partitions = int(5.0e2)

# Path in which model will be saved.
model_path = 'million_test.data'

# Coefficient determining how strong the model's regularization will be weighted. Higher regularization -> higher penalty for model complexity. 
regularization = 5.0e-4

# Amount of chess games to sample for the batch learning process.
downsample = 1.0e6

# Loss function used to train the model.
loss_function = 'log'
model_args = {'loss':loss_function,'alpha':regularization}

pipeline = ChessPipeline(pgn_directory,model_args)
pipeline.batch_learning(num_partitions,model_path,downsample)
```
Since holding the features for hundreds of thousands of games all at once will likely cause memory overflow, batch learning randomly partitions the different chess games found in the pgn files. Then, a batch of input features is made for each partition individually so that input features are needed for only one batch at a given time. This way, we can prevent memory overflow errors that would likely occur if we were to instead partition the entire input feature set (for all games). These game partitions are then iteratively processed into feature batches and used to train the ML model.

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
from ChessAI import ChessAI

cache_path = 'example_cache.data'
model_path = 'example_model.data'
ai = ChessAI(cache_path=cache_path,model_path=model_path)
```

If you have any questions or feedback, please email curiouscalvinj@gmail.com. Thanks for checking the project out.
