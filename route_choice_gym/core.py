from abc import ABC, abstractmethod


# An action of the agent
class Agent(ABC):
    def __init__(self):

        self.__OD_pair = ''

        self.__last_action = 0

        self.__flow = 1.0

        self.__time_flexibility = 0.5

    @abstractmethod
    def get_OD_pair(self) -> str:
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
