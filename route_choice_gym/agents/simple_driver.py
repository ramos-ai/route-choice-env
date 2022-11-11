from typing import Optional, Callable

from route_choice_gym.core import DriverAgent, Strategy

# Simple agent class for testing the environment
class SimpleDriver(DriverAgent):

    def __init__(self, od_pair: str, actions: list, strategy: Strategy):
        super(SimpleDriver, self).__init__()

        self.__od_pair = od_pair
        self.__last_action = 0

        self.__flow = 1.0
        self.__time_flexibility = 0.5

        # self.__strategy = {a: 0.0 for a in actions}
        self.__actions = actions
        self.__strategy = strategy

    def choose_action(self, obs):
        """
        :param obs: observation of the agent
        :return: returns an action
        """
        self.__last_action = self.__strategy.choose_action(obs)
        return self.__last_action

    def update_policy(self, obs_, reward):
        raise NotImplementedError

    def get_od_pair(self):
        return self.__od_pair

    def get_flow(self):
        return self.__flow

    def get_time_flexibility(self):
        return self.__time_flexibility

    def get_last_action(self):
        return self.__last_action
