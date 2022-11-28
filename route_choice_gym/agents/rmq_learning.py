
from route_choice_gym.core import DriverAgent, Policy


class RMQLearning(DriverAgent):

    def __init__(self, od_pair, actions, flow=1.0, extrapolate_costs=True, policy: Policy = None):
        super(RMQLearning, self).__init__()

        self.__od_pair = od_pair
        self.__actions = actions
        self.__last_action = -1
        self.__iteration = 0

        # Flow controlled by the agent (default is 1.0)
        self.__flow = flow

        # Strategy (Q table)
        self.__strategy = {a: 0.0 for a in self.__actions}

        # Sum of the experimented costs (used to obtain the average cost)
        self.__sum_cost = 0.0

        # Whether the average cost estimations should extrapolate the experimented costs
        self.__extrapolate_costs = extrapolate_costs

        # History
        # For each action, store an [sum, samples, extrapolated_sum, avg, last, last_time] array, where:
        # * sum:                the sum of costs experimented for this action (only considers the times when it is,
        #                       in fact, chosen)
        # * samples:            the number of samples composing the sum (the number of times the action was chosen),
        # * extrapolated_sum:   is an extrapolation of the sum of costs (when the action is not the chosen one,
        #                       then the sum is incremented with the last value, given in the next field) and
        # * avg:                is the average cost, which may be defined (using parameter self.__extrapolate_costs)
        #                       considering the sum or the extrapolated_sum
        # * last:               is the most updated cost of this action
        # * last_time:          is the most updated travel time of this action (useful for computing deltatolling)
        self.__history_actions_costs = {
            a: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for a in actions
        }

        # Estimated regret
        self.__estimated_regret = None
        self.__estimated_action_regret = {a: 0.0 for a in actions}

        # Real regret
        self.__real_regret = 0.0

        # Minimum average cost (stored here to enhance performance)
        self.__min_avg_cost = 0.0

        # Policy used for choosing an action
        self.__policy = policy

    def get_od_pair(self):
        return self.__od_pair

    def get_last_action(self):
        return self.__last_action

    def get_strategy(self):
        return self.__strategy

    def get_flow(self):
        return self.__flow

    # Calculate the driver's estimated regret
    def __estimate_regret(self):
        self.__estimated_regret = (self.__sum_cost / self.__iteration) - self.__min_avg_cost

        # Estimated regret per action
        for a in self.__history_actions_costs:
            if self.__extrapolate_costs is False and self.__history_actions_costs[a][1] == 0:  # just to handle initial cases
                self.__estimated_action_regret[a] = 0.0
            else:
                self.__estimated_action_regret[a] = self.__history_actions_costs[a][3] - self.__min_avg_cost

    # calculate the real regret given the real minimum average cost
    def update_real_regret(self, real_min_avg: float):
        self.__real_regret = self.get_average_cost() - real_min_avg

    def get_real_regret(self):
        return self.__real_regret

    def get_estimated_regret(self, action: int = None):
        if action is None:
            return self.__estimated_regret
        else:
            return self.__estimated_action_regret[action]

    def get_last_cost(self):
        return self.__history_actions_costs[self.__last_action][4]

    def get_average_cost(self):
        return self.__sum_cost / self.__iteration

    def choose_action(self, obs: int):

        # increment the iteration counter
        self.__iteration += 1
        self.__last_action = self.__policy.act(obs, d=self)
        return self.__last_action

    def update_strategy(self, obs_: int, cost: float, alpha: float = None, regret_as_cost: bool = False) -> None:
        """

        :param obs_:
        :param cost: for the route choice problem, reward is the cost of taking an action
        :param alpha: learning rate
        :param regret_as_cost: whether th agent should use regret as cost
        :return: None
        """

        # store the dictionary locally to improve performance
        self_hac = self.__history_actions_costs

        # update most recent travel time of current taken action (useful for computing deltatolling)
        self_hac[self.__last_action][5] = cost

        # update the sum of costs (used to compute the average cost)
        self.__sum_cost += cost

        # update the history of costs
        # Pt1: for the current action
        self_hac[self.__last_action][0] += cost  # add current cost
        self_hac[self.__last_action][1] += 1  # increment number of samples
        self_hac[self.__last_action][4] = cost  # update last cost
        # NOTE: position 5 in the array is updated earlier to keep the travel time (instead of cost)

        # Pt2: for all actions...
        self.__min_avg_cost = float('inf')
        for a in self_hac:

            # update the extrapolated sum
            self_hac[a][2] += self_hac[a][4]  # add last cost to extrapolate estimation

            # compute the average cost
            if self.__extrapolate_costs:
                avg_cost = self_hac[a][2] / self.__iteration
            else:
                try:  # just to handle initial cases
                    avg_cost = self_hac[a][0] / self_hac[a][1]
                except ZeroDivisionError:
                    avg_cost = 0

            self_hac[a][3] = avg_cost

            if avg_cost < self.__min_avg_cost:
                self.__min_avg_cost = avg_cost

        # Update the regret
        self.__estimate_regret()
        if regret_as_cost:
            cost = self.get_estimated_regret(self.__last_action)

        # Normalize cost (Reward)
        normalised_utility = 1 - cost

        # Update the strategy (Q table)
        self.__strategy[self.__last_action] = (1 - alpha) * self.__strategy[self.__last_action] + alpha * normalised_utility
