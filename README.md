# Chess-System
This project is about analyzing patterns from large amounts of chess game data. Specifically, this code can train supervised ML models from hundreds of thousands of chess games. These models can then be tested and experimented with by playing chess against an AI that uses these models for optimal play. More detailed instructions about installation and different features of the project are shown below.

## Table of Contents (Optional)

> If your `README` has a lot of info, section headers might be nice.

- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [Team](#team)
- [FAQ](#faq)
- [Support](#support)
- [License](#license)

## Installation 

### Clone
You can clone this repository by using the following link: BLANK. 

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
A pipeline for parsing and analyzing large amounts of pgn chess files. Pgn is a format in which chess games can be represented, and a pgn file is a file containing multiple pgn chess games. The pipeline is designed with certain ways in which it can analyze pgn chess positions. These include the amount of each player's piece types on the board, where each piece is located, bishop pairs, pawn development, etc.. However, you can include whatever methods you want to use for analysis easily as long as they are done so with python-chess: https://python-chess.readthedocs.io/en/latest/. 

Note that in order to find any meaningful insights about chess from game play, tens of thousands of expert games are very likely a bare minimum. The example training data provided in the directory *training_data* contains hundreds of thousands of expert games thanks to pgnmentor.com.

To get started with a ChessPipeline, you need to initialize it with two parameters: *pgn_directory* and *model_args*. The former is a directory in which pgn files are located; the latter consists of the desired ML model configurations. Note that the ML model is a *SGDClassifier*. Different models can be implemented without much modification to the code. Here is an example of how to start a ChessPipeline from pgn data in *training_data*, run batch learning on 200,000 of its games, and save the resulting model to *final_test.data*. 

```shell
$ from ChessPipeline import ChessPipeline

$ pgn_directory = 'training_data'
$ num_partitions = int(1.0e2)
$ model_path = 'final_test.data'
$ regularization = 5.0e-4
$ downsample = 2.0e5
$ loss_function = 'log'
$ model_args = {'loss':loss_function,'alpha':regularization}

pipeline = ChessPipeline(pgn_directory=pgn_directory,model_args=model_args)
pipeline.batch_learning(num_partitions,model_path,downsample)
```
Since holding the features for hundreds of thousands of games all at once will likely cause memory overflow, batch learning randomly partitions the chess positions found in the different games amongst the pgn files. Then, a batch of input features is made for each partition individually, which is then used for batch training. This way, memory is only needed for one batch at a time. 

### Chess AI
AI engine used for calculating opponent moves in the *ChessGame* module. It uses an alpha-beta pruning search to determine optimal moves given a valuation function. The valuation function assesses how good a chess position is for either player. You can choose to use either an ML valuation function or a heuristic based valuation function. Heuristic valuation can be customized by changing the *get_features* method. Input features used for heuristic/model-driven valuation can be customized through modifying *get_heuristic_features/get_model_features*. 

### Chess Game 
A Python interface for directly playing chess in Terminal. When playing chess games with *ChessGame.py*, opponent game play stems from an instance of *ChessAI*. You can use this to test ML models built from a chess pipeline; test any arbitrary chess evaluation function (for example, using heuristics or a ML model trained with *ChessPipeline*); or just play for fun.

Note that this script uses a position cache to store chess positions and their valuations as they're calculated. This means that when changing models, the path should be changed as well. Without doing so, different chess positions can be calculated with different valuation functions in the same game. The following excerpt is taken from the end of the *ChessGame* script. This is the code in which changes should be made to reflect new models.

``` shell
$ from ChessAI import ChessAI

$ cache_path = 'example_cache.data'
$ model_path = 'example_model.data'
$ ai = ChessAI(cache_path=cache_path,model_path=model_path)
```

If you have any questions or feedback, please email curiouscalvinj@gmail.com. Thanks for checking the project out.
