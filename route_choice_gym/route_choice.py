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

    def __init__(self, problem_instance: ProblemInstance, normalise_costs=True, agent_vehicles_factor=1.0):

        self.__normalize_costs = normalise_costs

        self.__problem_instance = problem_instance
        self.__problem_instance.reset_graph()

        self.__solution = self.__problem_instance.get_empty_solution()
        self.__solution_time_flexibility = self.__problem_instance.get_empty_solution()

        # agents of the environment
        self.drivers = []
        self.n_agents = 0

        # n_of_agents_per_od and action_space are both dictionary, mapping from OD pairs
        self.n_of_agents_per_od = {}
        self.action_space = Dict()
        # self.observation_space =  # TODO

        for od in self.__problem_instance.get_OD_pairs():
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
        Returns obs, reward, terminal for each agent.

        """
        obs_n = []
        reward_n = []
        terminal_n = []

        self.__solution = self.__problem_instance.get_empty_solution()

        count_drivers = {od: {} for od in self.__problem_instance.get_OD_pairs()}
        for i, d in enumerate(self.drivers):

            try:
                count_drivers[d.get_od_pair()][action_n[i]] += 1
            except:
                count_drivers[d.get_od_pair()][action_n[i]] = 0

            od_order = self.__problem_instance.get_OD_order(d.get_od_pair())
            self.__solution[od_order][action_n[i]] += d.get_flow()

        print(f"count_drivers: {count_drivers}")
        print(f"solution: {self.__solution}")
        self.__problem_instance.evaluate_assignment(self.__solution)

        for d in self.drivers:
            obs_n.append(self.__get_obs(d))
            reward_n.append(self.__get_reward(d))

            # receives True because of the stateless nature of the problem
            terminal_n.append(True)

        return obs_n, reward_n, terminal_n

    def reset(self, *, seed=None, options=None):
        self.__problem_instance.reset_graph()

        self.__solution = self.__problem_instance.get_empty_solution()

        obs_n = []
        for _ in self.drivers:
            obs_n.append(0.0)
        return obs_n

    def get_env_obs(self):
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
        reward = self.__get_cost(d)
        return reward

    def __get_cost(self, d):
        """
        :param d: Driver instance
        :return: route cost
        """
        route = self.__problem_instance.get_route(d.get_od_pair(), d.get_last_action())
        cost = route.get_cost(self.__normalize_costs)
        return cost
