import gym
from gym.spaces import Dict, Discrete

from decimal import Decimal
from typing import List

from route_choice_gym.core import DriverAgent
from route_choice_gym.problem import ProblemInstance


class RouteChoice(gym.Env):
    """
    Definitions
        obs:
            is the flow of vehicles on a route taken by the driver.

        reward:
            is the negative of the cost of a route.

    Structures
        act_n:
            an array from the size of n_agents, each index
            corresponds to the action chosen by each agent.

        obs_n:
            an array from the size of n_agents, each index
            corresponds to the observation of each agent.

        reward_n:
            an array from the size of n_agents, each index
            corresponds to the reward of each agent.

    """

    def __init__(self, problem_instance: ProblemInstance, agent_vehicles_factor=1.0, revenue_redistribution_rate=0.0,
                 normalise_costs=True, tolling=False):

        self.__problem_instance = problem_instance
        self.__problem_instance.reset_graph()

        self.__normalize_costs = normalise_costs
        self.__tolling = tolling

        self.__revenue_redistribution_rate = revenue_redistribution_rate

        self.__solution = list()
        self.__solution_w_preferences = list()

        # agents of the environment
        self.drivers = []
        self.n_agents = 0

        self.__avg_cost = 0.0
        self.__normalised_avg_cost = 0.0

        # n_of_agents_per_od and action_space are both dictionary, mapping from OD pairs
        self.n_of_agents_per_od = {}
        self.action_space = Dict()
        # self.observation_space =  # TODO

        for od in self.od_pairs:
            n_agents = int(Decimal(str(self.__problem_instance.get_OD_flow(od))) / Decimal(str(float(agent_vehicles_factor))))

            self.n_agents += n_agents
            self.n_of_agents_per_od[od] = n_agents
            self.action_space[od] = Discrete(self.__problem_instance.get_route_set_size(od))

            # Initial costs
            # initial_costs = []
            # for r in self.__problem_instance.get_routes(od):
            #     initial_costs.append(r.get_cost(self.__normalize_costs))

    def set_drivers(self, drivers: List[DriverAgent]):
        if isinstance(drivers[0], DriverAgent) and len(drivers) == self.n_agents:
            self.drivers = drivers

    def step(self, action_n):
        """
        This function makes a step in the environment. It receives an array of actions taken by the agents.

        We have two data structures for the solutions to evaluate an assignment, at every step we initiate those to
        empty solutions:

        - solution: it stores the flow of agents in every route taken by the agents.
                    we than use this information to add the flow to the routes/links to calculate the tt e cost.

        - solution_with_preferences: it stores flow of agents according to its preferences on the time-money trade-off.
                                     we use this information to help calculate marginal costs and tolls.

        After evaluating then assignment we return the obs_n, reward_n, terminal_n which are arrays mapping to each
        driver.
        """
        obs_n = []
        reward_n = []
        terminal_n = []

        self.__solution = self.__problem_instance.get_empty_solution()
        self.__solution_w_preferences = self.__problem_instance.get_empty_solution()

        for i, d in enumerate(self.drivers):
            od_order = self.__problem_instance.get_OD_order(d.get_od_pair())
            self.__solution[od_order][action_n[i]] += d.get_flow()
            self.__solution_w_preferences[od_order][action_n[i]] += d.get_flow() * (1 - d.get_preference_money_over_time())

        self.__avg_cost, self.__normalised_avg_cost = self.__problem_instance.evaluate_assignment(self.__solution, self.__solution_w_preferences)

        # if self.__tolling:

        #    if self.__revenue_redistribution_rate > 0.0:
        #        tolls_share_per_od = [0.0 for _ in range(len(self.od_pairs))]

        #    for d in self.drivers:

        #        route = self.__problem_instance.get_route(d.get_OD_pair(), d.get_last_action())

        #        # compute the cost
        #        cost = self.__get_cost(d)

        #        # toll-based methods required additional values to compute rewards:
        #        # weighted MCT needs the weighted marginal costs
        #        if self.__weighted_MCT:
        #            additional_cost = route.get_weighted_marginal_cost(self.__normalize_costs)
        #        # other methods need the free flow travel time
        #        else:
        #            additional_cost = route.get_free_flow_travel_time(self.__normalize_costs)

        #        # compute the toll
        #        toll = d.compute_toll_dues(cost, additional_cost)

        #        # update the total revenue
        #        if self.__revenue_redistribution_rate > 0.0:
        #            od = self.__problem_instance.get_OD_order(d.get_OD_pair())
        #            tolls_share_per_od[od] += toll

        #    # compute the share to be redistributed with the agents
        #    if self.__revenue_redistribution_rate > 0.0:
        #        for od in self.od_pairs:
        #            od_order: int = self.__problem_instance.get_OD_order(od)
        #            tolls_share_per_od[od_order] = (tolls_share_per_od[od_order] * self.__revenue_redistribution_rate) / self.__problem_instance.get_OD_flow(od)

        for d in self.drivers:
            obs_n.append(self.__get_obs(d))
            reward_n.append(self.__get_reward(d))
            terminal_n.append(True)  # receives True because of the stateless nature of the problem

        return obs_n, reward_n, terminal_n

    def reset(self, *, seed=None, options=None):
        self.__problem_instance.reset_graph()

        self.__solution = self.__problem_instance.get_empty_solution()
        self.__solution_w_preferences = self.__problem_instance.get_empty_solution()

        obs_n = []
        for _ in self.drivers:
            obs_n.append(0.0)
        return obs_n

    @property
    def avg_cost(self):
        return self.__avg_cost

    @property
    def od_pairs(self):
        return self.__problem_instance.get_OD_pairs()

    @property
    def problem_instance(self):
        return self.__problem_instance

    @property
    def solution(self):
        return self.__solution

    def __get_obs(self, d):
        """
        :param d: Driver instance
        :return: obs
        """
        od_order = self.__problem_instance.get_OD_order(d.get_od_pair())
        obs = self.__solution[od_order][d.get_last_action()]
        return obs

    def __get_reward(self, d):
        """
        :param d: Driver instance
        :return: reward on the route choice problem is the cost of taking a route
        """
        if self.__tolling:
            reward = (self.__get_cost(d), self.__get_toll_dues(d))
        else:
            reward = (self.__get_cost(d), 0.0)
        return reward

    def __get_cost(self, d):
        """
        :param d: Driver instance
        :return: route cost
        """
        route = self.__problem_instance.get_route(d.get_od_pair(), d.get_last_action())
        cost = route.get_cost(self.__normalize_costs)
        return cost

    def __get_toll_dues(self, d):
        """
        :param d: Driver instance
        :return: toll dues calculated by the driver
        """
        raise NotImplementedError
