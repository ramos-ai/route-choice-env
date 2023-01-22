

class Policy(object):
    """
        Interface for a generic policy.

        It defines functions to implement a policy for a driver agent.
    """

    def __init__(self):
        pass

    def act(self, d):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError


class TollingStrategy(object):

    @staticmethod
    def compute_toll(cost, additional_cost=0.0, time_flexibility=0.0):
        raise NotImplementedError

    @staticmethod
    def compute_cost(cost, additional_cost=0.0, time_flexibility=0.0, toll_dues=0.0):
        raise NotImplementedError


class DriverAgent(object):
    """
        Interface for a generic driver agent.

        It defines attributes and functions to implement an agent that interacts with the route choice environment.
    """

    __actions: list
    __flow: float
    __last_action: int
    __od_pair: str
    __preference_money_over_time: float
    __strategy: dict

    def get_flow(self) -> float:
        raise NotImplementedError

    def get_last_action(self) -> int:
        raise NotImplementedError

    def get_od_pair(self) -> str:
        raise NotImplementedError

    def get_preference_money_over_time(self) -> float:
        raise NotImplementedError

    def get_strategy(self):
        raise NotImplementedError

    def choose_action(self) -> int:
        raise NotImplementedError
