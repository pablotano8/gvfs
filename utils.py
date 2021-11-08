import numpy as np
import torch


def reward_to_go(rewards):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def format_input(pos, r_pos, size, gvf_net) -> np.array:
    return np.array([(pos[0] - size / 2) / 25, (pos[1] - size / 2) / 25,
           (r_pos[0] - size / 2) / 25, (r_pos[1] - size / 2) / 25] + \
           gvf_net.gvfs[:, pos[0], pos[1], :, :].flatten().tolist() + \
           gvf_net.Q_gamma[pos[0], pos[1], :, :].flatten().tolist())


def train_epoch(i, change, batch_size, env, gvf_net_xf, gvf_net_k1, gvf_net_k2, logits_net, optimizer, device, nb_episodes_random_policy, which_subgoal):

    #"which_subgoal" is one of the subgoals (k1,k2,xf)=(0,1,2)
    
    # shortcuts
    size = gvf_net_xf.size
    num_actions = gvf_net_xf.num_actions
    L = env.maze.objects.free.positions

    # make some empty lists for logging.
    batch_obs, batch_acts, batch_weights, \
                ep_rews, batch_rets, batch_lens = [], [], [], [], [], []

    # reset episode
    _, done = env.reset(), False
    h = torch.zeros(logits_net.hidden_dim, dtype=torch.float32).to(device) if logits_net.name == 'rnn' else None
    
    while True:
        pos = np.squeeze(env.maze.objects.agent.positions).copy()
        k1_pos = np.squeeze(env.maze.objects.k1.positions).copy()
        k2_pos = np.squeeze(env.maze.objects.k2.positions).copy()
        xf_pos = np.squeeze(env.maze.objects.xf.positions).copy()
        
        if which_subgoal==0:
            obs = format_input(pos, k1_pos, size, gvf_net_k1)
        elif which_subgoal==1:
            obs = format_input(pos, k2_pos, size, gvf_net_k2)
        elif which_subgoal==2:
            obs = format_input(pos, xf_pos, size, gvf_net_xf)
        batch_obs.append(obs)

        # if exploration phase is True move randomly
        if np.any([(i - k) % change == 0 for k in range(nb_episodes_random_policy)]):
            act = np.random.randint(num_actions)            
        # otherwise go to the randomly chosen subgoal (k1,k2,xf)=(0,1,2):
        else: 
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            act, h = logits_net.get_action(torch.as_tensor(obs, dtype=torch.float32).to(device), h)
        # move
        _, _, done, _ = env.step(act)

        # next position
        pos_next = np.squeeze(env.maze.objects.agent.positions).copy()

        # utility at next position
        utility_xf = np.all(pos_next == xf_pos) * 1.
        utility_k1 = np.all(pos_next == k1_pos) * 1.
        utility_k2 = np.all(pos_next == k2_pos) * 1.

        # update gvfs in exploratory phase
        if np.any([(i - k) % change == 0 for k in range(nb_episodes_random_policy)]):
            gvf_net_k1.update(pos, utility_k1, act, pos_next, L)
            gvf_net_k2.update(pos, utility_k2, act, pos_next, L)
            gvf_net_xf.update(pos, utility_xf, act, pos_next, L)
            
        # terminate if one of the subgoals was found (or at 200 steps):
        
        if which_subgoal==0 and utility_k1==1:
            rew = 50
            done = True
        elif which_subgoal==1 and utility_k2==1:
            rew = 50
            done = True
        elif which_subgoal==2 and utility_xf==1:
            rew = 50
            done = True            
        else:
            rew = -0.2
            
        # save action and reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done == True:
            # save cumulative reward and length of episode
            batch_rets.append(sum(ep_rews))
            batch_lens.append(len(ep_rews))

            # reward to go
            batch_weights += list(reward_to_go(ep_rews))
            batch_loss = 0
            
            # reset env
            _, done, ep_rews = env.reset(), False, []
            h = torch.zeros(logits_net.hidden_dim, dtype=torch.float32).to(device) if logits_net.name == 'rnn' else None
            if len(batch_obs) > batch_size:
                break
          
        
    if np.all([(i - k) % change != 0 for k in range(nb_episodes_random_policy)]):
        h = torch.zeros(logits_net.hidden_dim, dtype=torch.float32).to(device) if logits_net.name == 'rnn' else None
        optimizer.zero_grad()
        batch_loss = logits_net.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to(device),
                                             act=torch.as_tensor(batch_acts, dtype=torch.int32).to(device),
                                             weights=torch.as_tensor(batch_weights, dtype=torch.float32).to(device),
                                             lens=torch.as_tensor(batch_lens, dtype=torch.float32).to(device)
                                            )
        batch_loss.backward()
        optimizer.step()
        
    return batch_loss, batch_rets, batch_lens
