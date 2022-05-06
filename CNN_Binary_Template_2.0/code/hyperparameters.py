"""
    Hyperparameters for a run.
"""

parameters = {
    # Random Seed
    'seed': 123,

    # Data
    'train_data': '../dataset/train_data.csv',
    'test_data': '../dataset/test_data.csv',  # Path to the training and testing directories

    'img_size': 380,  # Image input size (this might change depending on the model)  # was 380
    'batch_size': 48,  # Input batch size for training (you can change this depending on your GPU ram)
    'data_mean': [0.3920153081417084, 0.3703818917274475, 0.3400942385196686],  # Mean values for each layer (RGB) (THIS CHANGE FOR EVERY DATASET)
    'data_std': [0.1949324756860733, 0.1925087571144104, 0.20403994619846344],  # Std Dev values for each layer (RGB) (THIS CHANGE FOR EVERY DATASET)
    'out_features': 1,  # For binary is 1

    # Model
    'model': 'efficientnet',  # Model to train (This name has to correspond to a model from models.py)
    'optimizer': 'ADAM',  # Optimizer to update model weights (Currently supported: ADAM or SGD)
    'criterion': 'BCEWithLogitsLoss',
    'lear_rate': 0.001,  # Learning Rate to use
    'min_epochs': 10,  # Minimum number of epochs to train for
    'epochs': 30,  # Number of epochs to train for
    'precision': 16,  # Pytorch precision in bits
    'accumulate_grad_batches': 4,  # the number of batches to estimate the gradient from
    'num_workers': 0  # Number of CPU workers to preload the dataset in parallel
}
