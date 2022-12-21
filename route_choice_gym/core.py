
class DriverAgent(object):
    def __init__(self):
        pass

    def get_od_pair(self) -> str:
        raise NotImplementedError

    def get_flow(self) -> float:
        raise NotImplementedError

    def get_preference_money_over_time(self) -> float:
        raise NotImplementedError

    def get_last_action(self) -> int:
        raise NotImplementedError

    def get_strategy(self):
        raise NotImplementedError


class Policy(object):
    def __init__(self):
        pass

    def act(self, obs, d):
        raise NotImplementedError

    def update_policy(self, **kwargs):
        raise NotImplementedError
