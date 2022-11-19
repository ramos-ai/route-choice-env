from abc import ABC, abstractmethod


# An action of the agent
class DriverAgent(ABC):
    __od_pair = ''
    __last_action = 0
    __flow = 1.0
    __time_flexibility = 0.5

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
    def action(self, obs):
        raise NotImplementedError

    @abstractmethod
    def update(self, obs_, reward, alpha):
        raise NotImplementedError
