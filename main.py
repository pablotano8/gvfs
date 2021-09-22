import torch
from policy_networks.mlp import PolicyNet
from torch import optim
from tasks import RandomMaze
import numpy as np
from tasks import Maze_config
from gvf_networks.tabular import Tabular
from utils import train_epoch

# get gpu when available
if torch.cuda.is_available():
    use_gpu = True
    device = torch.device("cuda:0")
    print("GPU is available", flush=True)
else:
    use_gpu = False
    device = torch.device("cpu")
    print("no GPU found", flush=True)

# parameters
gammas = [0.99]
num_actions = 4
lr = 5e-4
env_name = 'RandomMaze-v0'
size = 20
depth = 5
threshold_rew = 0.1
change = 50  # maze is changed every change epochs
batch_size = 1000
nb_epochs = 10000

# declare gvf network
gvf_net = Tabular(size=size, num_actions=num_actions, depth=depth, gammas=gammas, threshold_rew=threshold_rew)

# declare policy network
logits_net = PolicyNet(4 + (depth + 1) * num_actions * len(gammas), num_actions).to(device)

# make optimizer
optimizer = optim.Adam(logits_net.parameters(), lr=lr)

# define environment
env = RandomMaze.main(env_name=env_name, size=size)
L = env.maze.objects.free.positions
Maze_config.goal_idx = [L[np.random.randint(0, len(L))]]
Maze_config.start_idx = Maze_config.goal_idx

# first epoch
batch_loss, batch_rets, batch_lens = train_epoch(0, change, batch_size, env, gvf_net, logits_net, optimizer, device)