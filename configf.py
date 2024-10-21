import torch

config = {
    'C' : 2,
    'EPSILON' : 0.5,
    'ALPHA' : 0.9,
    'MCTS_SEARCHES' : 1000
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")