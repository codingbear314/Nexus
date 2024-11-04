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


import multiprocessing as mp

def worker_func(worker_id, 
        root, 
        request_List, 
        response_List, 
        end_flag,
        barrier_requesting,
        barrier_model_inference_ready,
        barrier_next_iteration_ready
    ):
    while not end_flag.is_set():
        node = root
        while node.expanded:
            node = node.select()
        
        isterminal = node.board.checkTerminal()[0]
        if isterminal:
            value = -1
            # Loss, but still we need to make the barrier, since it can block other processes
            request_List[worker_id] = None
            barrier_requesting.wait()
            barrier_model_inference_ready.wait()
        else:
            inp = node.board.get_board().unsqueeze(0).to(NN_device)
            legal_moves = node.board.get_legal_moves()
            request_List[worker_id] = inp
            barrier_requesting.wait()
            # So the response is requested, and processing
            # Note, it's not done.

            barrier_model_inference_ready.wait()
            # Now, the response is ready
            # So, we can get the response

            actor, critic = response_List[worker_id]
            actor.squeeze_() # Logically, this has to be 1d tensor, but just to be safe
            critic.squeeze_() # Logically, this has to be 0d tensor, but just to be safe
            actor = torch.softmax(actor, dim=0)
            actor = actor * legal_moves
            if actor.sum() > 0:
                actor = actor / actor.sum()
            else:
                actor = legal_moves / legal_moves.sum()
            
            node.expand(actor)
            value = critic.item()
        
        node.backpropagate(value)
        barrier_next_iteration_ready.wait()

class ParallelMCTS:
    def __init__(self, model):
        self.model = model
        self.model.to(NN_device)
        self.num_workers = config.MCTS_Workers
    
        
    def search(self, states):
        num_workers = min(self.num_workers, len(states))

        request_L = [None] * num_workers
        response_L = [None] * num_workers
        end_flag = mp.Event()

        barrier_requesting = mp.Barrier(num_workers + 1) # +1 for main process
        barrier_model_inference_ready = mp.Barrier(num_workers + 1) # +1 for main process
        barrier_next_iteration_ready = mp.Barrier(num_workers + 1) # +1 for main process

        barrier_requesting.reset()
        barrier_model_inference_ready.reset()
        barrier_next_iteration_ready.reset()

        # Share the memory
        request_L = mp.Manager().list(request_L)
        response_L = mp.Manager().list(response_L)

        processes = []
        roots = [Node(state.copy(), 0, None, None) for state in states]

        for i in range(num_workers):
            p = mp.Process(target=worker_func, args=(
                i, roots[i], request_L, response_L, end_flag, barrier_requesting, barrier_model_inference_ready, barrier_next_iteration_ready
            ))
            p.start()
            processes.append(p)
        
        for search_iteration in range(config.MCTS_Searches):
            barrier_requesting.wait()
            # Now, the requests are made
            # Now, we need to get the model output
            # inp = torch.stack(request_L, dim=0).to(NN_device)
            # We have to note that, request_L can have None, so we need to filter it
            inp = []
            indexes = []
            for i, req in enumerate(request_L):
                if req is not None:
                    inp.append(req)
                    indexes.append(i)
            inp = torch.stack(inp, dim=0).to(NN_device)
            actor, critic = self.model(inp)
            for i, idx in enumerate(indexes):
                response_L[idx] = (actor[i].detach().to(device), critic[i].detach().to(device))
                # Actor, critic shape is [225], [1] since we are indexing on batch dimension
            barrier_model_inference_ready.wait()
            # Now, the model inference is done
            # Now, we need to wait for the next iteration
            if search_iteration == config.MCTS_Searches - 1:
                end_flag.set()
            barrier_next_iteration_ready.wait()

        for p in processes:
            p.join()

        rtr_probs = torch.zeros(num_workers, 15 * 15, dtype=torch.float, device=device, requires_grad=False)
        for i, root in enumerate(roots):
            for action, child in enumerate(root.children):
                rtr_probs[i, action] = root.Nsa[action]
            rtr_probs[i] = rtr_probs[i] / rtr_probs[i].sum()
        return rtr_probs