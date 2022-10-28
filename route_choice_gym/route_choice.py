import gym
from gym.spaces import Dict, Discrete
from decimal import Decimal

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

    def __init__(self, P: ProblemInstance, normalise_costs=True, agent_vehicles_factor=1.0):

        self.NORMALISE_COSTS = normalise_costs

        self.P = P
        self.P.reset_graph()

        self.S, self.S_time_flexibility = self.P.get_empty_solution(), self.P.get_empty_solution()

        # agents of the environment
        self.drivers = []
        self.n_agents = 0

        # n_of_agents_per_od and action_space are both dictionary, mapping from OD pairs
        self.n_of_agents_per_od = {}
        self.action_space = Dict()
        # self.observation_space =  # TODO

        for od in self.P.get_OD_pairs():
            n_agents = int(Decimal(str(self.P.get_OD_flow(od))) / Decimal(str(float(agent_vehicles_factor))))

            self.n_agents += n_agents
            self.n_of_agents_per_od[od] = n_agents
            self.action_space[od] = Discrete(self.P.get_route_set_size(od))

            # Initial costs
            # initial_costs = []
            # for r in self.P.get_routes(od):
            #     initial_costs.append(r.get_cost(self.NORMALISE_COSTS))

    def step(self, action_n):
        """
        Returns obs, reward, terminal for each agent.

        """
        obs_n = []
        reward_n = []
        terminal_n = []

        self.S, self.S_time_flexibility = self.P.get_empty_solution(), self.P.get_empty_solution()

        for i, d in enumerate(self.drivers):
            od_order = self.P.get_OD_order(d.get_OD_pair())
            self.S[od_order][action_n[i]] += d.get_flow()
            self.S_time_flexibility[od_order][action_n[i]] += d.get_flow() * (1 - d.get_time_flexibility())

        print(f"solution: {self.S}")
        self.P.evaluate_assignment(self.S, self.S_time_flexibility)

        for d in self.drivers:
            obs_n.append(self._get_obs(d))
            reward_n.append(self._get_reward(d))
            terminal_n.append(False)

        return obs_n, reward_n, terminal_n

    def reset(self):
        self.P.reset_graph()
        self.S, self.S_time_flexibility = self.P.get_empty_solution(), self.P.get_empty_solution()

        obs_n = []
        for d in self.drivers:
            obs_n.append(self._get_obs(d))
        return obs_n

    def _get_obs(self, d):
        """
        :param d: Driver instance
        :return: obs
        """
        od_order = self.P.get_OD_order(d.get_OD_pair())
        obs = self.S[od_order][d.get_last_action()]
        return obs

    def _get_reward(self, d):
        """
        :param d: Driver instance
        :return: reward
        """
        reward = self._get_cost(d)
        return -reward

    def _get_cost(self, d):
        """
        :param d: Driver instance
        :return: route cost
        """
        route = self.P.get_route(d.get_OD_pair(), d.get_last_action())
        cost = route.get_cost(self.NORMALISE_COSTS)
        return cost

    def get_env_obs(self):
        return self.S
