from route_choice_env.core import Agent, Policy


# Simple agent class for testing the environment
class SimpleDriver(Agent):

    def __init__(
            self,
            actions: list,
            d_id: str,
            policy: Policy
            ):

        self.__driver_id = d_id
        self.__actions = actions
        self.__last_action = 0

        # For SimpleDriver, the strategy is same for all actions and does not update
        self.__strategy = {a: 1.0 for a in self.__actions}

        self.__policy = policy

    # -- Driver Properties
    # ----------------------------------------
    def get_last_action(self):
        return self.__last_action

    def get_strategy(self):
        return self.__strategy

    # -- Agent functions
    # ----------------------------------------
    def choose_action(self):
        """
        :return: returns an action
        """
        self.__last_action = self.__policy.act(d=self)
        return self.__last_action
