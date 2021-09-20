import torch
from policy_networks.mlp import PolicyNet
from torch import optim
from tasks import RandomMaze
import numpy as np
from tasks import Maze_config
from gvf_networks.tabular import Tabular
from policy_networks.net import reward_to_go

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


def train_epoch(i):
    # make some empty lists for logging.
    batch_obs, batch_acts, batch_weights, \
                ep_rews, batch_rets, batch_lens = [], [], [], [], [], []

    # reset episode
    _, done = env.reset(), False

    while True:
        pos = np.squeeze(env.maze.objects.agent.positions).copy()
        r_pos = np.squeeze(env.maze.objects.goal.positions).copy()

        obs = np.array([(pos[0] - size / 2) / 25, (pos[1] - size / 2) / 25,
                       (r_pos[0] - size / 2) / 25, (r_pos[1] - size / 2) / 25] + \
                       gvf_net.gvfs[1:, pos[0], pos[1], :, :].flatten().tolist() + \
                       gvf_net.Q_gamma[pos[0], pos[1], :, :].flatten().tolist())

        # save obs
        batch_obs.append(obs)

        # take action
        if np.any([(i - k) % change == 0 for k in range(12)]):
            act = np.random.randint(num_actions)
        else:
            act = logits_net.get_action(torch.as_tensor(obs, dtype=torch.float32).to(device))

        # get reward
        _, rew, done, _ = env.step(act)

        # next position
        pos_next = np.squeeze(env.maze.objects.agent.positions).copy()

        # utility at next position
        utility = np.all(pos_next == r_pos) * 1.

        # update gvfs
        gvf_net.update(pos, utility, act, pos_next, L)

        # rescale rewards
        rew = 50 if rew == 1 else -0.2

        # save action and reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            # save cumulative reward and length of episode
            batch_rets.append(sum(ep_rews))
            batch_lens.append(len(ep_rews))

            # reward to go
            batch_weights += list(reward_to_go(ep_rews))

            # reset env
            _, done, ep_rews = env.reset(), False, []

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

        batch_loss = 0
        if np.all([(i - k) % change != 0 for k in range(12)]):
            print(i)
            optimizer.zero_grad()
            batch_loss = logits_net.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to(device),
                                                 act=torch.as_tensor(batch_acts, dtype=torch.int32).to(device),
                                                 weights=torch.as_tensor(batch_weights, dtype=torch.float32).to(device)
                                                 )
            batch_loss.backward()
            optimizer.step()

    return batch_loss, batch_rets, batch_lens