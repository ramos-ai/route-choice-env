import gym
from gym.spaces import Dict, Discrete, Space

from decimal import Decimal
from typing import List

from route_choice_env.core import DriverAgent
from route_choice_env.problem import Network


class RouteChoice(gym.Env):
    """
    Definitions
        obs:
            is None due to the problem being a stateless MDP

        reward:
            is the travel cost experienced by the agent

        info:
            dictionary containing info about the route taken by the agent:
            {
                free_flow_travel_time
            }

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

        info_n:
            an array from the size of n_agents, each index
            corresponds to the info each agent has access.
    """

    def __init__(
            self,
            road_network: Network,
            agent_vehicles_factor=1.0,
            revenue_redistribution_rate=0.0,
            normalise_costs=True,
            tolling=False
    ):

        self.__road_network = road_network
        self.__road_network.reset_graph()

        self.__normalize_costs = normalise_costs
        self.__tolling = tolling

        self.__revenue_redistribution_rate = revenue_redistribution_rate
        self.__tolls_share_per_od = []

        self.__solution = list()
        self.__solution_w_preferences = list()

        self.__avg_cost = 0.0
        self.__normalised_avg_cost = 0.0

        # sum of routes' costs along time (used to compute the averages)
        self.routes_costs_sum = {od: [0.0 for _ in range(self.__road_network.get_route_set_size(od))] for od in self.od_pairs}
        self.routes_costs_min = {od: 0.0 for od in self.od_pairs}

        # env spaces
        self.observation_space = Space(None)
        self.action_space = Dict()  # action space is mapped to OD pairs
        self.n_agents_per_od = {}

        # env agents
        self.drivers = []
        self.n_agents = 0
        for od in self.od_pairs:
            n_agents = int(Decimal(str(self.__road_network.get_OD_flow(od))) / Decimal(str(float(agent_vehicles_factor))))

            self.n_agents += n_agents
            self.n_agents_per_od[od] = n_agents
            self.action_space[od] = Discrete(self.__road_network.get_route_set_size(od))

        self.__iteration = 0

    @property
    def avg_travel_time(self):
        return self.__avg_travel_time

    @property
    def od_pairs(self):
        return self.__road_network.get_OD_pairs()

    @property
    def road_network(self):
        return self.__road_network

    @property
    def solution(self):
        return self.__solution

    def get_free_flow_travel_times(self, od):
        return [r.get_cost(self.__normalize_costs) for r in self.__road_network.get_routes(od)]

    def get_route_set_size(self, od):
        return self.__road_network.get_route_set_size(od)

    def set_drivers(self, drivers: List[DriverAgent]):
        if isinstance(drivers[0], DriverAgent) and len(drivers) == self.n_agents:
            self.drivers = drivers

    def __update_routes_costs_stats(self):
        for od in self.od_pairs:
            for r in range(int(self.__road_network.get_route_set_size(od))):
                cc = self.__road_network.get_route(od, r).get_cost(True)
                if self.__tolling:
                    cc = 2 * cc - self.__road_network.get_route(od, r).get_free_flow_travel_time(self.__normalize_costs)
                self.routes_costs_sum[od][r] += cc
            self.routes_costs_min[od] = min(self.routes_costs_sum[od]) / (self.__iteration + 1)

    def step(self, action_n: list):
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
        info_n = []

        self.__solution = self.__road_network.get_empty_solution()
        self.__solution_w_preferences = self.__road_network.get_empty_solution()

        # Evaluate solution based on routes taken and flow of drivers
        for i, d in enumerate(self.drivers):
            od_order = self.__road_network.get_OD_order(d.get_od_pair())
            self.__solution[od_order][action_n[i]] += d.get_flow()
            self.__solution_w_preferences[od_order][action_n[i]] += d.get_flow() * (1 - d.get_preference_money_over_time())

        self.__avg_travel_time, self.__normalised_avg_travel_time = self.__road_network.evaluate_assignment(self.__solution, self.__solution_w_preferences)

        # Update the sum of routes' costs (used to compute the averages)
        self.__update_routes_costs_stats()

        for d in self.drivers:
            obs_n.append(None)
            reward_n.append(self.__get_reward(d))
            terminal_n.append(True)  # receives True because of the stateless nature of the problem
            info_n.append(self.__get_info(d))

        self.__iteration += 1

        return obs_n, reward_n, terminal_n, info_n

    def reset(self, *, seed=None, options=None):
        self.__road_network.reset_graph()

        self.__solution = self.__road_network.get_empty_solution()
        self.__solution_w_preferences = self.__road_network.get_empty_solution()

        obs_n = []
        info_n = []
        for d in self.drivers:
            obs_n.append(None)
            info_n.append(self.__get_info(d))
        return obs_n, info_n

    def __get_obs(self, d):
        """
        Observation of the agent is None due to the problem being a stateless MDP.

        :param d: Driver instance
        :return: obs
        """
        return None

    def __get_reward(self, d):
        """
        Reward is the experienced travel cost by the agent.

        :param d: Driver instance
        :return: reward
        """
        travel_cost = self.__get_travel_cost(d)
        return travel_cost

    def __get_info(self, d):
        """
        Info has some information about the route taken by the agent.

        :param d: Driver instance
        :return: obs
        """

        route = self.__road_network.get_route(d.get_od_pair(), d.get_last_action())
        info = {
            "free_flow_travel_time": route.get_free_flow_travel_time(self.__normalize_costs)
        }
        return info

    def __get_travel_cost(self, d):
        """
        :param d: Driver instance
        :return: driver's travel time
        """
        route = self.__road_network.get_route(d.get_od_pair(), d.get_last_action())
        cost = route.get_cost(self.__normalize_costs)
        return cost
