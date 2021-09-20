import torch.nn as nn
from policy_networks.net import Net


class PolicyNet(Net):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        self.fc = nn.Sequential(

            nn.Linear(input_shape, 50),
            nn.ReLU(),
            nn.Linear(50, n_actions)
        )

    def forward(self, x):
        return self.fc(x)
