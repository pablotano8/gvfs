# imports
import numpy as np
import gym
from tasks.Env import Env
from tasks import Maze_config

if __name__ == '__main__':
    Maze_config.goal_idx = [[1, 11]]
    Maze_config.start_idx = [[11, 6]]
    Maze_config.x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                              [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                              [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                              [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                              [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    env_name = 'RandomMaze-v0'
    gym.envs.register(id=env_name, entry_point=Env, max_episode_steps=200)
    env = gym.make(env_name)
    obs = env.reset()
    import matplotlib.pyplot as plt
    plt.imshow(obs)
    plt.show()
    # obs, rew, done, _ = env.step(0) # take a step
    # env.maze.objects.goal.positions[0] # get position
