import torch.nn as nn
from policy_networks.net import Net


class PolicyNet(Net):
    def __init__(self, input_shape, n_actions, num_layers=2, nb_units=[64, 64]):
        super(Net, self).__init__()
        # if nb_units is integer create a list of size num_layers
        if isinstance(nb_units, int):
            nb_units = [nb_units] * num_layers

        if len(nb_units) != num_layers:
            raise ValueError('Length of number of nb_units is not num_layers')

        # create network
        network = [nn.Linear(input_shape, nb_units[0]),
                   nn.ReLU()]
        for i_layer in range(1, num_layers):
            network += [nn.Linear(nb_units[i_layer - 1], nb_units[i_layer]),
                        nn.ReLU()]
        network += [nn.Linear(nb_units[-1], n_actions)]

        self.fc = nn.Sequential(*network)

    def forward(self, x):
        return self.fc(x)
