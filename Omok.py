import numpy
import random
import torch
from device import device

class OmokGame:
    def __init__(self):
        self.board = torch.zeros(15, 15, dtype=torch.float, device=device, requires_grad=False)
        self.turn = 1
        self.move_history = []

    def reset(self):
        self.board.zero_()
        self.turn = 1
        self.move_history = []
    
    def get_board(self):
        return self.board
    
    def get_legal_moves(self):
        rtr = torch.ones(15 * 15, dtype=torch.float, device=device, requires_grad=False)
        for move in self.move_history:
            rtr[move[0] * 15 + move[1]] = 0
        return rtr
    
    def get_turn(self):
        return self.turn
    
    def makeMove(self, x, y):
        if self.board[x, y] != 0:
            return False
        self.board[x, y] = self.turn
        self.move_history.append((x, y))
        self.turn = -self.turn
        return True
    
    def checkWin(self):
        lastMove = self.move_history[-1]
        # Search fron last move
        def UtilitySqCheck(x, y):
            if x < 0 or x >= 15 or y < 0 or y >= 15:
                return 0
            return self.board[x, y]
        
        last_Player = self.board[lastMove[0], lastMove[1]]

        dxl = [-1, 0, 1, -1, 1, -1, 0, 1]
        dyl = [-1, -1, -1, 0, 0, 1, 1, 1]

        for i in range(8):
            count = 1
            for j in range(1, 5):
                if UtilitySqCheck(lastMove[0] + dxl[i] * j, lastMove[1] + dyl[i] * j) == last_Player:
                    count += 1
                else:
                    break
            for j in range(1, 5):
                if UtilitySqCheck(lastMove[0] - dxl[i] * j, lastMove[1] - dyl[i] * j) == last_Player:
                    count += 1
                else:
                    break
        
            if count == 5:
                return last_Player
        return 0
    
    def play(self, x, y):
        if not self.makeMove(x, y):
            return False
        return self.checkWin()
    
    def __str__(self):
        ans = ""
        for i in range(15):
            for j in range(15):
                if self.board[i, j] == 1:
                    ans += "X"
                elif self.board[i, j] == -1:
                    ans += "O"
                else:
                    ans += "."
            ans += "\n"
        return ans
    
    def __repr__(self):
        return "Project Nexus : Omok Game"

    def copy(self):
        game = OmokGame()
        game.board = self.board.clone()
        game.turn = self.turn
        game.move_history = self.move_history.copy()
        return game

if __name__ == '__main__':
    game = OmokGame()
    while True:
        x, y = map(int, input().split())
        print(game.play(x, y))
        print(game)