from chesspipeline import ChessPipeline

# Path that the pipeline will use for saving/loading its ML model. 
model_path = 'chess_rfc_example.data'
pipeline = ChessPipeline(model_path)

# Build training and testing feature/result batches using the games found 
# in the pgn files from <train_batches_dir> and <test_batches_dir>.
train_batch_dir = 'train_batches'
num_batch_files = 10
batch_size = int(1.0e3)
train_dir = 'train_pgns'
max_batches = 10
reset_batches = False
pipeline.build_batches(train_dir, train_batch_dir, num_batch_files, reset_batches,
					   batch_size, max_batches)

test_dir = 'test_pgns'
test_batch_dir = 'test_batches'
pipeline.build_batches(test_dir, test_batch_dir, num_batch_files, reset_batches, 
					   batch_size, max_batches)

# Grid search over different values for different random forest hyperparameters.
# Training and testing is done with batch learning using some of the feature/output
# batches built above. 
param_grid = {'max_depth': [40, 80], 'min_samples_leaf': [100, 1000]}
batch_sizes = [int(x) for x in [1.0e5, 1.0e6]]
grid_results = pipeline.grid_search(train_batch_dir, test_batch_dir, param_grid, 
								    batch_sizes, max_train_batches=2, max_test_batches=1)

print(f"Grid search results with batch size = {batch_size}: {grid_results}")
opt_params, opt_batch_size = grid_results[0][0:2]
print(f"Optimal hyperparameters: {opt_params}")
pipeline.update_model_params(opt_params)
pipeline.batch_learning(train_batch_dir, opt_batch_size)