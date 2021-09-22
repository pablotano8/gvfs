import numpy as np
import torch
from policy_networks.net import reward_to_go


def train_epoch(i, change, batch_size, env, gvf_net, logits_net, optimizer, device):
    # shortcuts
    size = gvf_net.size
    num_actions = gvf_net.num_actions
    L = env.maze.objects.free.positions

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