import torch.nn as nn
from policy_networks.net import Net
from torch.distributions.categorical import Categorical
import torch
from torch.nn.utils.rnn import pad_sequence


class RNNPolicyNet(Net):
    def __init__(self, input_shape, n_actions, num_layers=1, nb_units=64):
        super(Net, self).__init__()
        self.name = 'rnn'

        # Number of hidden dimensions
        self.hidden_dim = nb_units

        # Number of hidden layers
        self.layer_dim = num_layers

        # RNN
        self.rnn = nn.RNN(input_shape, self.hidden_dim, self.layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, n_actions)

    def forward(self, x, hn):
        out, hn = self.rnn(x, hn)
        out = self.fc(out)
        return out, hn

    def get_policy(self, obs, h):
        # create new axis of obs corresponding to (batch_size and seq_length)
        # create new axis of hn corresponding to (num_layers and batch_size)
        if len(obs.shape) == 1:
            logits, h = self.forward(obs[None, None], h[None, None])
        else:
            logits, h = self.forward(obs, h)
        return Categorical(logits=torch.squeeze(logits)), torch.squeeze(h)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(self, obs, h):
        categorical, h = self.get_policy(obs, h)
        return categorical.sample().item(), h

    def compute_loss(self, obs, act, weights, lens):
        assert (torch.sum(lens) == obs.shape[0])

        ind = torch.cat((torch.zeros(1), torch.cumsum(lens, 0)))

        obs_reshaped = torch.transpose(
                            pad_sequence([obs[int(ind[i]):int(ind[i+1])] for i in torch.arange(len(ind) - 1)]), 0, 1
        )
        act_reshaped = torch.transpose(
                            pad_sequence([act[int(ind[i]):int(ind[i+1])] for i in torch.arange(len(ind) - 1)]), 0, 1
        )

        h = torch.zeros([1, len(obs_reshaped), self.hidden_dim])
        categorical, _ = self.get_policy(obs_reshaped, h)

        logp = categorical.log_prob(act_reshaped)
        logp_ravel = torch.cat([logp[i, :int(lens[i])] for i in torch.arange(len(lens))])

        return -(logp_ravel * weights).mean()
