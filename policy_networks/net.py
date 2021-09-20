from abc import ABC
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def reward_to_go(rewards):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


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
