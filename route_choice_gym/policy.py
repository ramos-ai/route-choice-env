import numpy as np
import random

from route_choice_gym.core import DriverAgent, Policy


class Random(Policy):

    def __init__(self, actions: list):
        super(Random, self).__init__()

        self.__actions = actions

    def act(self, obs, d: DriverAgent):
        return random.choice(self.__actions)

    def update_policy(self):
        pass


class EpsilonGreedy(Policy):
    """
        Generic epsilon greedy policy class.

        Act method receives a Driver Agent and selects an action from its strategy (which is generally the q-table).
    """

    def __init__(self, epsilon: float, min_epsilon: float = 0.0):
        super(EpsilonGreedy, self).__init__()

        self.__epsilon = epsilon
        self.__min_epsilon = min_epsilon

    def act(self, obs, d: DriverAgent):

        # Epsilon-greedy: choose the action with highest probability with probability 1-epsilon
        # otherwise, choose any action uniformly at random
        if np.random.random() < self.__epsilon:
            return int(np.random.random() * len(d.get_strategy()))  # slightly slower than random.random, but it is less biased
        else:
            return max(d.get_strategy(), key=d.get_strategy().get)

    def update_policy(self, epsilon_decay: float = 0.99):
        if self.__epsilon > self.__min_epsilon:
            self.__epsilon = self.__epsilon * epsilon_decay
        else:
            self.__epsilon = self.__min_epsilon
