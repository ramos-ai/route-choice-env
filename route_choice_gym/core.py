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
        raise NotImplementedError

    @abstractmethod
    def get_flow(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_time_flexibility(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_last_action(self) -> int:
        raise NotImplementedError


class Strategy(ABC):
    def __init__(self, actions):
        pass

    @abstractmethod
    def choose_action(self):
        raise NotImplementedError

    @abstractmethod
    def update_strategy():
        raise NotImplementedError
