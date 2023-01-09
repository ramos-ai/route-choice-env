from route_choice_gym.core import DriverAgent, Policy


# Simple agent class for testing the environment
class SimpleDriver(DriverAgent):

    def __init__(self, od_pair: str, actions: list, policy: Policy):
        super(SimpleDriver, self).__init__()

        self.__od_pair = od_pair
        self.__last_action = 0

        self.__flow = 1.0
        self.__time_flexibility = 0.5

        # self.__strategy = {a: 0.0 for a in actions}
        self.__actions = actions
        self.__policy = policy

    # -- Driver Properties
    # ----------------------------------------
    def get_flow(self):
        return self.__flow

    def get_last_action(self):
        return self.__last_action

    def get_od_pair(self):
        return self.__od_pair

    def get_preference_money_over_time(self):
        return self.__preference_money_over_time

    def get_strategy(self):
        return self.__strategy

    def choose_action(self, obs):
        """
        :param obs: observation of the agent
        :return: returns an action
        """
        self.__last_action = self.__policy.act(obs, d=self)
        return self.__last_action
