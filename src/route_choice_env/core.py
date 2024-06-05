
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


class Agent(object):
    """
        Interface for a generic driver agent.

        It defines attributes and functions to implement an agent that interacts with the route choice environment.
    """
    __driver_id: str
    __actions: list
    __last_action: int
    __strategy: dict

    def get_last_action(self):
        raise NotImplementedError

    def get_strategy(self):
        raise NotImplementedError

    def choose_action():
        raise NotImplementedError


class Driver:
    """
        Class describing adriver in the environment.

        It is instantiated in the PettingZoo env __init__ function.
    """
    __d_id: str
    __od_pair: str
    __flow: float
    __preference_money_over_time: float
    current_route: int

    def __init__(
            self,
            d_id: str,
            od_pair: str,
            flow: float = 1.0,
            preference_money_over_time: float = 0.5
    ):

        self.__d_id = d_id
        self.__od_pair = od_pair

        # Flow controlled by the agent (default is 1.0)
        self.__flow = flow

        # The time-money trade-off (the higher the preference of money over time is, the more the agent prefers
        # saving money or, alternatively, the less it cares about travelling fast)
        self.__preference_money_over_time = preference_money_over_time

        self.current_route = -1

    # -- Getter
    # ------------
    def get_flow(self) -> float:
        return self.__flow

    def get_id(self) -> str:
        return self.__d_id

    def get_od_pair(self) -> str:
        return self.__od_pair

    def get_preference_money_over_time(self) -> float:
        return self.__preference_money_over_time

    def get_current_route(self):
        return self.current_route

    # -- Setter
    # ------------
    def set_current_route(self, route_id: int):
        self.current_route = route_id
