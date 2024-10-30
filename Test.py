import numpy
import random
import torch
import math

from Node import Node
from Game import NexusGame
from MCTS import MCTS
from Model import Nexus_Small
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = Nexus_Small().to(device)
    
    mcts = MCTS(model)

    game = NexusGame()
    game.play_move(0)
    game.play_move(1)
    game.play_move(15)
    print(game)

    probs = mcts.search(game)
    print(probs)

if __name__ == '__main__':
    main()