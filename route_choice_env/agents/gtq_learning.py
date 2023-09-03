from route_choice_env.core import Agent, Policy


class GTQLearning(Agent):  # Implementation of Regret Minimisation Q-Learning

    def __init__(self,
                 actions,
                 d_id: str,
                 extrapolate_costs=False,
                 policy: Policy = None
                 ):

        self.__driver_id = d_id
        self.__actions = actions
        self.__last_action = -1

        # Strategy (Q table)
        self.__strategy = {a: 0.0 for a in self.__actions}

        # Policy used for choosing an action
        self.__policy = policy

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

        # Whether the average cost estimations should extrapolate the experimented costs
        self.__extrapolate_costs = extrapolate_costs

        # Sum of the experimented costs (used to obtain the average cost)
        self.__sum_cost = 0.0

        # Current toll dues
        self.__toll_dues = 0.0

        # Minimum average cost (stored here to enhance performance)
        self.__min_avg_cost = 0.0

        self.__iteration = 0

    # -- Driver Properties
    # ----------------------------------------
    def get_last_action(self):
        return self.__last_action

    def get_strategy(self):
        return self.__strategy

    def get_average_cost(self):
        return self.__sum_cost / self.__iteration

    # -- Agent functions
    # ----------------------------------------
    def choose_action(self):
        self.__iteration += 1
        self.__last_action = self.__policy.act(d=self)
        return self.__last_action

    def update_strategy(self, obs_: tuple, reward: float, info: dict, alpha: float = None) -> None:
        """
        This function does 4 things:
        - computes the tolls for the agent using travel_time and free_flow_travel_time of route chosen
        - compute agent's utility as travel_time + computed tolls
        - updates agent's strategy (Q-table) using utility
        - updates agent's history

        :param
            obs_: None
            reward: travel_time of route chosen
            alpha: learning rate
            info: dict with: free_flow_travel_time

        :return: None
        """
        travel_time = reward
        driver_preference = info['driver_preference']
        route_free_flow_tt = info['free_flow_travel_times'][self.__last_action]
        side_payment = info['side_payments'][self.__last_action]

        # Compute toll dues
        self.compute_toll_dues(travel_time, route_free_flow_tt, driver_preference)

        # Compute utility (cost) and update strategy (Q-table)
        cost = self.__compute_utility(travel_time, side_payment)
        self.__update_strategy_q_learning(cost, alpha)

        # Update agent history
        self.__update_history(cost, travel_time)

    def __compute_utility(self, travel_time: float, side_payment: float) -> float:
        """
        For the TQLearningDriver algorithm, we compute the utility (cost) considering travel time and toll and use it as
        learning signal.

        :return: cost (reward)
        """
        return travel_time + self.__toll_dues - side_payment

    # Q-learning (stateless, so the gamma parameter is not required)
    def __update_strategy_q_learning(self, utility, alpha):
        normalised_utility = 1 - utility
        self.__strategy[self.__last_action] = (1 - alpha) * self.__strategy[self.__last_action] + alpha * normalised_utility

    # History (we keen an internal history of agent actions, useful for strategy and computing reward)
    def __update_history(self, cost: float, travel_time: float):
        # store the dictionary locally to improve performance
        self_hac = self.__history_actions_costs

        # update the sum of costs (used to compute the average cost)
        self.__sum_cost += cost

        # update the history of costs
        # Pt1: for the current action
        self_hac[self.__last_action][0] += cost  # add current cost
        self_hac[self.__last_action][1] += 1  # increment number of samples
        self_hac[self.__last_action][4] = cost  # update last cost
        self_hac[self.__last_action][5] = travel_time  # update most recent travel time of current taken action

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

    # -- Tolling functions
    # ----------------------------------------
    def get_toll_dues(self):
        return self.__toll_dues

    def compute_toll_dues(self,
                          travel_time: float,
                          free_flow_travel_time: float,
                          preference_money_over_time: float
                          ):
        # MCT with preferences
        # Note 1: this is equivalent to the original MCT if the preference parameter is 0.5 for all agents
        # Note 2: the expression is multiplied by 2 to make it fully compatible with the original MCT formulation
        #         (and to make the code retro-compatible with previous algorithms and validations)
        toll_mct = travel_time - free_flow_travel_time  # marginal cost
        toll_mct_w_preferences = (toll_mct + travel_time * preference_money_over_time)
        self.__toll_dues = toll_mct_w_preferences / preference_money_over_time  # the indifferent MCT
        return self.__toll_dues
