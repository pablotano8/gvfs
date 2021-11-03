from abc import ABC
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from utils import format_input


class Net(nn.Module, ABC):

    def __init__(self):
        super(nn.Module, self).__init__()

    # make function to compute action distribution
    def get_policy(self, obs, h):
        logits = self.forward(obs)
        return Categorical(logits=logits), h

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(self, obs, h=None):
        categorical, h = self.get_policy(obs, h)
        return categorical.sample().item(), h

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(self, obs, act, weights, lens=None):
        categorical, _ = self.get_policy(obs, None)
        logp = categorical.log_prob(act)
        return -(logp * weights).mean()

    # get policy for all states
    def get_policy_all_states(self, env, gvf_net, which_subgoal, device) -> np.ndarray:
        size = gvf_net.size
        policy_all_states = np.zeros([size, size, gvf_net.num_actions])
        h = torch.zeros(self.hidden_dim, dtype=torch.float32).to(device) if self.name == 'rnn' else None
        for x_pos in range(size):
            for y_pos in range(size):
                pos = np.array([x_pos, y_pos])
                if which_subgoal==0:
                    r_pos = np.squeeze(env.maze.objects.k1.positions).copy()
                elif which_subgoal==1:
                    r_pos = np.squeeze(env.maze.objects.k2.positions).copy()
                elif which_subgoal==2:
                    r_pos = np.squeeze(env.maze.objects.xf.positions).copy()
                obs = torch.as_tensor(format_input(pos, r_pos, size, gvf_net), dtype=torch.float).to(device)
                categorical, _ = self.get_policy(obs, h)
                policy_all_states[x_pos, y_pos] = categorical.probs.cpu().detach().numpy()
        return policy_all_states
