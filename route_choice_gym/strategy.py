import numpy as np
import random

from route_choice_gym.core import Strategy


class Random(Strategy):
    
    def __init__(self, actions: list):
        super(Random, self).__init__(actions)

        self.__actions = actions

    def choose_action(self, obs):
        return random.choice(self.__actions)

    def update_strategy(self, obs, reward):
        raise NotImplementedError


class EpsilonGreedy(Strategy):
    
    def __init__(self, actions: list, epsilon: float):
        super(EpsilonGreedy, self).__init__()

        self.__actions = actions
        self.__epsilon = epsilon
        self.__strategy = {a: 0.0 for a in self.__actions}

    def choose_action(self, obs):
        if np.random.random() < self.__epsilon:
            return int(np.random.random() * len(self.__strategy))
        else:
            return max(self.__strategy, key=self.__strategy.get)

    def update_strategy(self, obs, reward):
        raise NotImplementedError
