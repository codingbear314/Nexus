from MCTS import MCTS
from Node import Node
from Game import NexusGame
from Model import Nexus_Small
import config
import torch
import math
import time

from tqdm.auto import trange


device = torch.device("cpu")
NN_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NexusAgent:
    def __init__(self):
        self.model = Nexus_Small().to(NN_device)
        # self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()
        self.mcts = MCTS(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.Learning_Rate)
    
    def selfPlay(self):
        memory = []
        game = NexusGame()

        self.model.eval()
        while True:
            probs = self.mcts.search(game)
            action = torch.multinomial(probs, 1).item()
            game.play_move(action)

            isTerminal = game.checkTerminal()[0]
            if isTerminal:
                # The player won
                returnMemory = []
                for stateHist, probsHist, playerHist in memory:
                    outcome = 1 if playerHist == game.turn else -1
                    returnMemory.append((stateHist, probsHist, outcome))
                return returnMemory
            
            memory.append((game.get_board().clone(), probs, game.turn))
    
    def train(self, memory):
        self.model.train()
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), config.Batch_Size):
            batch = memory[batchIdx:min(batchIdx + config.Batch_Size, len(memory))]
            states, probs, outcomes = zip(*batch)
            states = torch.stack(states).to(NN_device)
            probs = torch.stack(probs).to(NN_device)
            outcomes = torch.tensor(outcomes, dtype=torch.float, device=NN_device)
            
            output_policy, output_value = self.model(states)

            loss_policy = torch.nn.functional.cross_entropy(output_policy, probs)
            loss_value = torch.nn.functional.mse_loss(output_value.squeeze(), outcomes)
            loss = loss_policy + loss_value

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
    def learn_by_self_play(self):
        for i in trange(config.Self_Play_Iters):
            memory = []
            for _ in trange(config.Self_Play_Games):
                memory += self.selfPlay()
            for _ in trange(config.Train_Epochs):
                self.train(memory)
        
        torch.save(self.model.state_dict(), "model.pth")

if __name__ == '__main__':
    agent = NexusAgent()
    agent.learn_by_self_play()