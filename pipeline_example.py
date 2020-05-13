from chesspipeline import ChessPipeline

# pgn_directory is the directory containing pgn files. 
pgn_directory = 'training_data'
num_partitions = int(5.0e2)

# Path in which model will be saved.
model_path = 'example_model.data'

# Coefficient determining how strong the model's regularization will be weighted. Higher regularization -> higher penalty for model complexity. 
regularization = 5.0e-4

# Amount of chess games to sample for the batch learning process.
downsample = 2.0e5

# Loss function used to train the model.
loss_function = 'log'
model_args = {'loss':loss_function,'alpha':regularization}

pipeline = ChessPipeline(pgn_directory,model_args)
pipeline.batch_learning(num_partitions,model_path,downsample)
