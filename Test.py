import numpy
import random
import torch
import math

import multiprocessing as mp

import time

from Node import Node
from Game import NexusGame
from MCTS import MCTS
from Model import Nexus_Small
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def worker_func_worker(worker_id, model, game):
    mcts = MCTS(model)
    return mcts.search(game)

def main():
    model = Nexus_Small().to(device)
    
    mcts = MCTS(model)

    gameList = []

    gme = NexusGame()
    gameList.append(gme)
    gme.play_move(1)
    gameList.append(gme)
    gme.play_move(2)
    gameList.append(gme)
    gme.play_move(3)
    gameList.append(gme)
    gme.play_move(4)
    gameList.append(gme)
    gme.play_move(5)
    gameList.append(gme)
    gme.play_move(6)
    gameList.append(gme)
    gme.play_move(7)
    gameList.append(gme)


    st = time.time_ns()
    processes = []

    # Multiprocessing for parallel MCTS
    for i in range(4):
        p = mp.Process(target=worker_func_worker, args=(i, model, gameList[i]))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    et = time.time_ns()
    print("Time taken: ", (et - st) / 1e9)

    

if __name__ == '__main__':
    main()