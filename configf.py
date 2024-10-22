import torch

config = {
    'C' : 2,
    'EPSILON' : 0.5,
    'ALPHA' : 0.9,
    'MCTS_SEARCHES' : 1000,
    'LearningRate' : 0.001,
    'TEMPERATURE' : 0.8,
    'BatchSize' : 64,
    'Iterations' : 10,
    'GamesPerIteration' : 128,
    'Epochs' : 10,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")