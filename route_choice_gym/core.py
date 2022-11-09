from abc import ABC, abstractmethod


# An action of the agent
class DriverAgent(ABC):
    def __init__(self):

        self.__od_pair = ''

        self.__last_action = 0

        self.__flow = 1.0

        self.__time_flexibility = 0.5

    @abstractmethod
    def get_od_pair(self) -> str:
        raise NotImplemented

    @abstractmethod
    def get_flow(self) -> float:
        raise NotImplemented

    @abstractmethod
    def get_time_flexibility(self) -> float:
        raise NotImplemented

    @abstractmethod
    def get_last_action(self) -> int:
        raise NotImplemented


class Strategy(ABC):
    def __init__(self, actions):
        self.__actions = actions

    @abstractmethod
    def choose_action(self):
        raise NotImplemented
