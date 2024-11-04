import numpy
import random
import torch
import math

from Node import Node
from Game import NexusGame
import config

device = torch.device("cpu")
NN_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MCTS:
    def __init__(self, model):
        self.model = model
        self.model.to(NN_device)
    
    @torch.inference_mode()
    def search(self, state : NexusGame):
        root = Node(state.copy(), 0, None, None)

        for search_iteration in range(config.MCTS_Searches):
            node = root
            while node.expanded:
                node = node.select()
            
            isterminal = node.board.checkTerminal()[0]
            
            value = -1 # If it's terminal, it's a loss, since the opponent played the last move
            
            if not isterminal:
                inp = node.board.get_board().unsqueeze(0).to(NN_device)
                actor, critic = self.model(inp)
                actor = torch.softmax(actor, axis=1).squeeze(0).detach().to(device)
                critic = critic.item()

                legal_moves = node.board.get_legal_moves()
                actor = actor * legal_moves
                actor = actor / actor.sum()

                node.expand(actor)

                value = critic

            node.backpropagate(value)
        
        rtr_probs = torch.zeros(15 * 15, dtype=torch.float, device=device, requires_grad=False)
        for action, child in enumerate(root.children):
            if child is not None:
                rtr_probs[action] = child.Total_visits
        rtr_probs = rtr_probs / rtr_probs.sum()
        return rtr_probs


# import torch.multiprocessing as mp 
# class ParallelMCTS:
#     def __init__(self, model):
#         self.model = model
#         self.model.to(NN_device)
#         self.num_workers = config.MCTS_Workers

    
#     @torch.inference_mode()
#     def search(self, state): # state is a list of NexusGame
#         num_workers = len(state)

#         def worker_func(worker_id, input_shared, output_shared, barrier):
#             while True:
#                 barrier.wait()
#                 if input_shared[worker_id] is None:
#                     break
#                 state = input_shared[worker_id]
#                 root = Node(state.copy(), 0, None, None)
                
#                 while root.expanded:
#                     node = root.select()
                
                