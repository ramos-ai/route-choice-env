import gym
from gym.spaces import MultiDiscrete
from decimal import Decimal
import numpy as np


class RouteChoice(gym.Env):

    def __init__(self, P, alpha=0.5, epsilon=1.0, alpha_decay=0.99, epsilon_decay=0.99, min_alpha=0.0, min_epsilon=0.0,
                 normalise_costs=True, regret_as_cost=True, agent_vehicles_factor=1.0, print_OD_pairs_every_episode=True):

        self.NORMALISE_COSTS = normalise_costs

        self.P = P
        self.S, self.S_time_flexibility = self.P.get_empty_solution(), self.P.get_empty_solution()
        self.drivers = []

        self.n_of_agents = []
        for od in self.P.get_OD_pairs():
            self.n_of_agents.append(int(Decimal(str(self.P.get_OD_flow(od))) / Decimal(str(float(agent_vehicles_factor)))))
            initial_costs = []
            for r in self.P.get_routes(od):
                initial_costs.append(r.get_cost(self.NORMALISE_COSTS))

        self.action_space = MultiDiscrete([self.P.get_route_set_size(od) for od in self.P.get_OD_pairs()])
        # self.observation_space =  # TODO

    def step(self, action_n):
        """
        Returns obs, reward, terminal, info for each agent.

        """
        obs_n = self.__get_obs()
        reward_n = []
        terminal_n = []

        for i, d in enumerate(self.drivers):
            od = d.get_OD_index()
            self.S[od][action_n[i]] += d.get_flow()
            self.S_time_flexibility[od][action_n[i]] += d.get_flow() * (1 - d.get_time_flexibility())

            reward_n.append(self.__get_cost(d, self.NORMALISE_COSTS))
            terminal_n.append(False)

        return obs_n, reward_n, terminal_n

    def reset(self):
        obs_n = self.__get_obs()
        return obs_n

    def __get_cost(self, d, NORMALISE_COSTS):
        cost = 0.0
        od = self.P.get_OD_pairs()[d.get_OD_index()]
        route = self.P.get_route(od, d.get_last_action())
        cost = route.get_cost(NORMALISE_COSTS)
        return cost

    def __get_obs(self):
        """
            TODO
        """
        return [0.0 for _ in self.drivers]
