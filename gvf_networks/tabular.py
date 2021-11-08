import numpy as np


class Tabular:
    def __init__(self, size, num_actions=4, depth=6, nb_subgoals_in_state=0,
                 gammas=[0.98],
                 lr={'sr': 0.3, 'gvfs': 0.6, 'Q': 0.1},
                 thresholds={'sr': 0.6, 'gvfs': 0.1}):
        self.size = size
        self.num_actions = num_actions
        self.depth = depth
        self.gammas = gammas
        self.thresholds = thresholds
        self.num_gammas = len(self.gammas)
        self.lr = lr
        self.nb_subgoals_in_state = nb_subgoals_in_state

        self.Q_gamma = np.zeros((2,) * nb_subgoals_in_state + (size, size, self.num_gammas, self.num_actions))

        if depth > 0:
            # sr + gvfs
            self.sr = np.zeros((2,) * nb_subgoals_in_state + (size, size, self.num_gammas))
            self.gvfs = np.zeros(
                (depth,) + (2,) * nb_subgoals_in_state + (size, size, self.num_gammas, self.num_actions))

    def update(self, state, utility, act, state_next, L, pi_behav=0.25, pi_targets=None):
        '''
        :param state: the state describing the environment to input the gvfs
        :param utility:
        :param act:
        :param state_next:
        :param L: available locations
        :param pi_behav: behavioral policy of size (4) <- gives the probability of each action
        :param pi_targets: vector should be None in the case of navigational GVFs. For structural GVFs,
        and a vector of size (nb_navigational_policies) = (3) in the 2 keys maze task and gives the
        probability of action=act and the state under navigational policies.
        :return:
        '''
        if ((pi_behav != 0.25) and pi_targets is None) or \
                (pi_behav is None and pi_targets is None):
            raise ValueError('pi_behav should be 0.25 and pi_targets None (case of exploratory phase for navigational'
                             ' GVFs learning) or both should be specified (case of structural GVFs learning)')
        # importance sampling weight for off-policy TD learning
        if pi_targets is None:
            IS_weights = np.ones(self.num_actions)
        else:
            IS_weights = pi_targets / pi_behav
            if len(IS_weights) != self.num_actions:
                raise ValueError('This is a mismatch in the dimension of pi_targets and self.num_actions')
        if np.any(np.isnan(IS_weights)):
            raise ValueError('is_weight is NaN')
        if np.any(IS_weights != 1) and self.nb_subgoals_in_state == 0:
            raise ValueError('off-policy TD is not developed yet for navigational GVFs')
        if np.all(IS_weights == 1) and self.nb_subgoals_in_state > 0:
            raise ValueError('pi_behav and pi_targets must be defined for structural GVFs')
        for g in range(self.num_gammas):
            self.Q_gamma[tuple(state) + (g, act)] = self.Q_gamma[tuple(state) + (g, act)] + \
                                                    self.lr['Q'] * (
                                                            utility + (1 - utility) * self.gammas[g] *
                                                            np.max(
                                                                self.Q_gamma[tuple(state_next) + (g,)]
                                                            )
                                                            - self.Q_gamma[tuple(state) + (g, act)]
                                                    )
            if self.depth > 0:
                # successor representation under the default policy
                has_sr = 1 if (self.nb_subgoals_in_state == 0) else 0

                self.sr[tuple(state) + (g,)] = self.sr[tuple(state) + (g,)] \
                                               + self.lr['sr'] * (utility + (1 - utility) * 0.25 / pi_behav * has_sr *
                                                                  self.gammas[g] * self.sr[tuple(state_next) + (g,)]
                                                                  - self.sr[tuple(state) + (g,)]
                                                                  )
                for i_depth in range(self.depth):
                    if self.nb_subgoals_in_state == 0:
                        assert (self.num_actions == 4), 'navigational gvfs not implemented in the case nb_actions != 4'
                        iterate_obj = range(self.num_actions), state + np.array([[1, 0], [-1, 0], [0, -1], [0, 1]]), \
                                      np.arange(2, 4)[None] - (np.arange(4)[:, None] > 1) * 2, IS_weights
                    else:
                        iterate_obj = np.arange(self.num_actions), np.tile(state_next[None], (self.num_actions, 1)), \
                                      np.vstack([np.delete(np.arange(self.num_actions), k)[None]
                                                 for k in range(self.num_actions)]), IS_weights
                        # take out IS
                        iterate_obj = [k[iterate_obj[-1] == 1] for k in iterate_obj]

                    # iterate_obj = gvf_id, next_pos, orth_gfvs, is_weight
                    for i_act, p, i_c, is_w in zip(*iterate_obj):
                        if np.any(np.all(L == p[-2:], axis=1)):
                            cumulant = (self.sr[tuple(p) + (g,)] > self.thresholds['sr']
                                        > self.sr[tuple(state) + (g,)]
                                        ) * 1 if (i_depth == 0) else \
                                np.any(self.gvfs[(i_depth - 1,) + tuple(p) + (g, i_c)] > self.thresholds['gvfs']) * \
                                ((not has_sr) or np.all(self.gvfs[(range(i_depth),) + tuple(state) + (g,)] < self.thresholds['gvfs'])) * \
                                (self.sr[tuple(state) + (g,)] < self.thresholds['sr']) * 1

                            self.gvfs[(i_depth,) + tuple(state) + (g, i_act)] = self.gvfs[
                                                                                    (i_depth,) + tuple(state) + (
                                                                                    g, i_act)] \
                                                                                + self.lr['gvfs'] * \
                                                                                (cumulant + (1 - cumulant) * is_w *
                                                                                 self.gammas[g] * self.gvfs[
                                                                                     (i_depth,) + tuple(p) + (g, i_act)]
                                                                                 - self.gvfs[
                                                                                     (i_depth,) + tuple(state) + (
                                                                                     g, i_act)]
                                                                                 )

    def reset(self):
        self.Q_gamma[:] = 0

        if self.depth > 0:
            self.sr[:] = 0
            self.gvfs[:] = 0


if __name__ == '__main__':
    gammas = [0.99]
    num_gammas = len(gammas)
