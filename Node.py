import numpy
import random
import torch
import math

import config
import Game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Node:
    def __init__(self, board, prior, action_t=None, parent=None):
        self.board = board
        self.parent = parent
        self.children = [None] * 225 

        self.legal_moves = None

        self.Total_visits = 0
        self.Total_value = 0
        self.ActorProb = prior

        self.action_taken = action_t

        self.ChildValueSum = torch.zeros(225, device=device)
        self.Nsa = torch.zeros(225, device=device)
        self.Psa = torch.zeros(225, device=device)

        self.expanded = False

    def select(self):
        if not self.expanded:
            return None

        Qsa_NA = torch.where(self.Nsa > 0, self.ChildValueSum / self.Nsa, 0.0)

        sqrt_parent = math.sqrt(self.Total_visits)
        UpperConfidence = 1 - (Qsa_NA + 1) / 2 + \
                          config.MCTS_C * self.Psa * sqrt_parent / (1 + self.Nsa)

        best_child_idx = torch.argmax(UpperConfidence).item()
        return self.children[best_child_idx]

    def expand(self, policy):
        have_children = False
        if self.legal_moves is None:
            self.legal_moves = self.board.get_legal_moves()
        for i in range(225):
            if self.legal_moves[i] > 0.5: # Due to floating point error, just be safe
                have_children = True
                new_game = self.board.copy()
                new_game.makeMove(i // 15, i % 15)
                self.children[i] = Node(new_game, policy[i], i, self)
                self.Psa[i] = policy[i]
        self.expanded = have_children

    def backpropagate(self, value):
        self.Total_visits += 1
        self.Total_value += value

        if self.parent is not None:
            self.parent.ChildValueSum[self.action_taken] += value
            self.parent.Nsa[self.action_taken] += 1
            self.parent.backpropagate(-value)