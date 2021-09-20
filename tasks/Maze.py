# imports
import numpy as np
from mazelab import BaseMaze, Object
from mazelab import DeepMindColor as Color
from tasks import Maze_config

class Maze(BaseMaze):

    def __init__(self):
        super().__init__()

    @property
    def size(self):
        return Maze_config.x.shape

    def make_objects(self):
        global x
        free = Object('free', 0, Color.free, False, np.stack(np.where(Maze_config.x == 0), axis=1))
        obstacle = Object('obstacle', 1, Color.obstacle, True, np.stack(np.where(Maze_config.x == 1), axis=1))
        agent = Object('agent', 2, Color.agent, False, [])
        goal = Object('goal', 3, Color.goal, False, [])
        return free, obstacle, agent, goal
