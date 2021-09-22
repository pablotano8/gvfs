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
    def get_policy(self, obs):
        logits = self.forward(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # get policy for all states
    def get_policy_all_states(self, env, gvf_net) -> np.ndarray:
        size = gvf_net.size
        policy_all_states = np.zeros([size, size, gvf_net.num_actions])
        for x_pos in range(size):
            for y_pos in range(size):
                pos = np.array([x_pos, y_pos])
                r_pos = np.squeeze(env.maze.objects.goal.positions).copy()
                obs = torch.as_tensor(format_input(pos, r_pos, size, gvf_net), dtype=torch.float)
                policy_all_states[x_pos, y_pos] = self.get_policy(obs).probs.detach().numpy()
        return policy_all_states
