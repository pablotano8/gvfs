# imports
import numpy as np
from mazelab import BaseEnv, VonNeumannMotion
from gym.spaces import Box, Discrete
from tasks.Maze import Maze
from tasks import Maze_config


class Env(BaseEnv):
    def __init__(self):
        super().__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_k1(current_position):
            self.has_k1 = True
            
        if self._is_k2(current_position):
            self.has_k2 = True
            
        if self._is_xf(current_position) and self.has_k2 == True and self.has_k1 == True:
            reward = +1
            done = True
            self.has_k1, self.has_k2 = False, False
            return self.maze.to_value(), [True, True, reward], done, {}
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self.maze.to_value(), [self.has_k1,self.has_k2, reward], done, {}

    def reset(self):
        self.maze = Maze()
        self.maze.objects.agent.positions = Maze_config.start_idx
        self.maze.objects.xf.positions = Maze_config.xf_idx
        self.maze.objects.k1.positions = Maze_config.k1_idx
        self.maze.objects.k2.positions = Maze_config.k2_idx
        self.has_k1, self.has_k2 = False, False
        return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable
    
    def _is_xf(self, position):
        out = False
        for pos in self.maze.objects.xf.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    
    def _is_k1(self, position):
        out = False
        for pos in self.maze.objects.k1.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    
    def _is_k2(self, position):
        out = False
        for pos in self.maze.objects.k2.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
