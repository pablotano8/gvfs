# imports
import numpy as np
from mazelab import BaseMaze, Object
from mazelab import DeepMindColor as Color
from tasks import Maze_config
import scipy

class Maze(BaseMaze):

    def __init__(self):
        super().__init__()

    @property
    def size(self):
        return Maze_config.x.shape

    def make_objects(self):
        k1 = Object('k1', 3, [0.2,0.5,0.8], False, [])
        k2 = Object('k2', 4, [0.8,0.5,0.2],  False, [])
        xf = Object('xf', 5, Color.goal ,False, [])
        free = Object('free', 0, Color.free, False, np.stack(np.where(Maze_config.x == 0), axis=1))
        obstacle = Object('obstacle', 1, Color.obstacle, True, np.stack(np.where(Maze_config.x == 1), axis=1))
        agent = Object('agent', 2, Color.agent, False, [])
        
        return free, obstacle, agent, k1, k2, xf
