# imports
import numpy as np
import gym
from tasks.Env import Env
from tasks import Maze_config
from mazelab.generators import random_maze


def main(start_position: str = 'goal_position', env_name: str = 'RandomMaze-v0', size: int = 20) -> object:
    if start_position not in ['goal_position']:
        print('For the moment, when a new maze is initialized, the agent is placed at the reward')
        raise NotImplementedError('This is not implemented yet')
    gym.envs.register(id=env_name, entry_point=Env, max_episode_steps=200)
    Maze_config.x = random_maze(width=size, height=size, complexity=1, density=0.5)
    env = gym.make(env_name)
    L = env.maze.objects.free.positions
    Maze_config.goal_idx = [L[np.random.randint(0, len(L))]]
    if start_position == 'goal_position':
        Maze_config.start_idx = Maze_config.goal_idx
    return env, L


if __name__ == '__main__':
    env, L = main(start_position='goal_position')