import numpy as np

from route_choice_env.core import Agent, Policy


class Random(Policy):
    """ Chooses action at random. """

    def __init__(self):
        super(Random, self).__init__()

    def act(self, d: Agent):
        return int(np.random.random() * len(d.get_strategy()))  # slower than random.random, but less biased

    def update(self):
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

    def act(self, d: Agent):

        # Epsilon-greedy: choose the action with the highest probability with probability 1-epsilon
        # otherwise, choose any action uniformly at random
        if np.random.random() < self.__epsilon:
            return int(np.random.random() * len(d.get_strategy()))  # slower than random.random, but less biased
        else:
            return max(d.get_strategy(), key=d.get_strategy().get)

    def update(self, epsilon_decay: float = 0.99):
        if self.__epsilon > self.__min_epsilon:
            self.__epsilon = self.__epsilon * epsilon_decay
        else:
            self.__epsilon = self.__min_epsilon
