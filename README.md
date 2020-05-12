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

ii) ChessAI.py: AI engine used by the ChessGame module. 

iii) ChessGame.py: A Python interface for playing chess that runs in terminal. You can use this to test ML models built from the chess pipeline, test any arbitrary chess evaluation function (for example, using heuristics), or simply play for fun. 

More detailed instructions are below. 

If you have any questions or feedback, please email me at curiouscalvinj@gmail.com. 
