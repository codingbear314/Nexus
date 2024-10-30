import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Nexus_Small(nn.Module):
    def __init__(self):
        super(Nexus_Small, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.Actor = nn.Linear(64, 225)
        self.Critic = nn.Linear(64, 1)
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(-1, 64)
        return self.Actor(x), torch.tanh(self.Critic(x))