# imports
import numpy as np
import gym
from tasks.Env import Env
from tasks import Maze_config
from mazelab.generators import random_maze


def main(env_name='RandomMaze-v0', size=20):
    gym.envs.register(id=env_name, entry_point=Env, max_episode_steps=200)
    Maze_config.x = random_maze(width=size, height=size, complexity=1, density=0.5)
    return gym.make(env_name)


if __name__ == '__main__':
    env = main()