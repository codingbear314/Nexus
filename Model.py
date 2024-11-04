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

class BottleneckBlock(nn.Module):
    expansion = 4  # Expansion factor for the output channels

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        # 1x1 Convolution (Reduction)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 Convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 Convolution (Expansion)
        out = self.conv3(out)
        out = self.bn3(out)

        # Add the shortcut
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out

class Nexus_Std(nn.Module):
    def __init__(self):
        super(Nexus_Std, self).__init__()
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottleneck layers
        self.layer1 = self._make_layer(BottleneckBlock, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BottleneckBlock, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(BottleneckBlock, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, 512, num_blocks=3, stride=2)

        # Global average pooling and fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Actor = nn.Linear(512 * BottleneckBlock.expansion, 225)
        self.Critic = nn.Linear(512 * BottleneckBlock.expansion, 1)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels * block.expansion, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels * block.expansion))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Add batch dimension if missing
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Bottleneck layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and output layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        actor_output = self.Actor(x)
        critic_output = torch.tanh(self.Critic(x))

        return actor_output, critic_output

if __name__ == "__main__":
    model = Nexus_Small()
    model.to(device)
    model.eval()
    print(model)
    x = torch.randn(7, 1, 15, 15).to(device)
    actor, critic = model(x)
    print(actor.shape, critic.shape)

    parameters = 0
    for p in model.parameters():
        parameters += p.numel()
    print("Total parameters: ", parameters)