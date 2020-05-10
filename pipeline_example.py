from ChessPipeline import ChessPipeline
from sklearn.metrics import mean_squared_error

pgn_directory = 'training_data'
num_partitions = int(1.0e3)
model_path = 'chess_test.data'
regularization = 5.0e-4
downsample = 2.0e5
loss_function = 'log'
model_args = {'loss':loss_function,'alpha':regularization}
pipeline = ChessPipeline(pgn_directory,model_args)
pipeline.batch_learning(num_partitions,model_path,downsample)
