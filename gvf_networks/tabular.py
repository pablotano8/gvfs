import numpy as np


class Tabular:
    def __init__(self, size, num_actions=4, depth=6, gammas=[0.99], threshold_rew=0.1):
        self.size = size
        self.num_actions = num_actions
        self.depth = depth
        self.gammas = gammas
        self.threshold_rew = threshold_rew
        self.num_gammas = len(self.gammas)
        self.lr = 0.1

        self.Q_gamma = np.zeros((size, size, self.num_gammas, self.num_actions))

        if depth > 0:
            # sr + gvfs
            self.sr = np.zeros([size, size, self.num_gammas])
            self.gvfs = np.zeros([depth, size, size, self.num_gammas, self.num_actions])

    def update(self, pos, utility, act, pos_next, L):
        for g in range(self.num_gammas):
            self.Q_gamma[pos[0], pos[1], g, act] = self.Q_gamma[pos[0], pos[1], g, act] + \
                                                   self.lr * (
                                                           utility + (1 - utility) * self.gammas[g] *
                                                           np.max(
                                                               self.Q_gamma[pos_next[0], pos_next[1], g, :]
                                                           )
                                                           - self.Q_gamma[pos[0], pos[1], g, act]
                                                   )
            if self.depth > 0:
                self.sr[pos[0], pos[1], g] = self.sr[pos[0], pos[1], g] \
                                             + 0.3 * (utility + (1 - utility) * self.gammas[g] * self.sr[
                    pos_next[0], pos_next[1], g] - self.sr[pos[0], pos[1], g]
                                                      )
                for i_depth in range(self.depth):
                    for i_act, p in enumerate(pos + np.array([[1, 0], [-1, 0], [0, -1], [0, 1]])):
                        if np.any(np.all(L == p, axis=1)):
                            i_c = np.arange(2, 4) - (i_act > 1) * 2
                            cumulant = (self.sr[p[0], p[1], g] > self.threshold_rew
                                        > self.sr[pos[0], pos[1], g]
                                        ) * 1 if (i_depth == 0) else \
                                np.any(self.gvfs[i_depth - 1, p[0], p[1], g, i_c] > self.threshold_rew) * \
                                np.all(self.gvfs[i_depth - 1, pos[0], pos[1], g, :] < self.threshold_rew)

                            self.gvfs[i_depth, pos[0], pos[1], g, i_act] = self.gvfs[
                                                                               i_depth, pos[0], pos[
                                                                                   1], g, i_act] + 0.8 * \
                                                                           (cumulant + (1 - cumulant) * self.gammas[g]
                                                                            * self.gvfs[
                                                                                i_depth, p[0], p[1], g, i_act]
                                                                            - self.gvfs[
                                                                                i_depth, pos[0], pos[1], g, i_act]
                                                                            )


if __name__ == '__main__':
    gammas = [0.99]
    num_gammas = len(gammas)
