from chesspipeline import ChessPipeline

# <pgn_directory_train> is the directory containing pgn files for training the model. 
pgn_directory_train = 'training_data'

# Path in which model will be saved. 
model_path = 'test_model.data'

# The number of partitions into which the pgn files will be split. The number of batches used during batch learning on each partition. 
num_partitions = 15
num_batches = 1000

# Coefficient determining how strong the model's regularization will be weighted. Higher regularization -> higher penalty for model complexity. 
regularization = 1.0e-3

# The portion of each batch to be used for training the model. Conversely, the remaining portion of the batch is used for later testing.  
train_size = 0.85

# Loss function used to train the model.
loss_function = 'log'
model_args = {'loss':loss_function,'alpha':regularization}

# Train model with batch learning on pgn files found in <pgn_directory_train>.
pipeline = ChessPipeline(pgn_directory_train,model_args)
validation_error = pipeline.batch_validation(num_partitions,num_batches,train_size=train_size,model_path=model_path)
print('Validation error: ' + str(validation_error))

# Test the above model on pgn test data found in <pgn_test_directory>. 
test_dir = 'test_data'
pipeline = ChessPipeline(pgn_directory,model_args,model_path=model_path)
test_error = pipeline.test_model(test_dir,num_partitions,num_batches)
print('Test error: ' + str(test_error))

