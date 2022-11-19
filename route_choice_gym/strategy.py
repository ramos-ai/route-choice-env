import numpy as np
import random

from route_choice_gym.core import Strategy


class Random(Strategy):

    def __init__(self, actions: list):
        super(Random, self).__init__(actions)

        self.__actions = actions

    def action(self, obs):
        return random.choice(self.__actions)

    def update(self, obs_, reward, alpha):
        return


class EpsilonGreedy(Strategy):

    def __init__(self, actions: list, epsilon: float):
        super(EpsilonGreedy, self).__init__()

        self.__actions = actions
        self.__epsilon = epsilon
        self.__last_action = None
        self.__strategy = {a: 0.0 for a in self.__actions}

    def action(self, obs):

        # epsilon-greedy: choose the action with highest probability with probability 1-epsilon
        # otherwise, choose any action uniformly at random
        if np.random.random() < self.__epsilon:
            self.__last_action = int(np.random.random() * len(self.__strategy))  # slightly slower than random.random, but it is less biased
        else:
            self.__last_action = max(self.__strategy, key=self.__strategy.get)

        return self.__last_action

    def update(self, obs_, reward, alpha=1.0):
        self.__strategy[self.__last_action] = (1 - alpha) * self.__strategy[self.__last_action] + alpha * reward
