import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Space

import functools
from decimal import Decimal
from typing import List, Optional

from route_choice_env.core import DriverAgent, EnvDriverAgent
from route_choice_env.problem import Network

from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn, AgentID


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

        # sum of routes' costs through time (used to compute the averages)
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


class RouteChoicePZ(ParallelEnv):
    """
        PettingZoo parallel environment implementation

        params:
            net_name: Name of the road network, maps the ./route_choice_env/networks folder
            routes_per_od: Number of routes per od.
            agent_vehicles_factor: Number of vehicles controlled by an agent of the environment.
            normalise_costs: Weather it should normalise its costs.

        __init__:
            - Create the road network and reset the graph.
            - Create the drivers for each OD (flow is defined in the <network>.net file).
            - Defines agents, observation_spaces, action_spaces.
            - Define other environment variables.

        step:
            -

        reset:
    """
    def __init__(
            self,
            net_name: str,
            routes_per_od: int,
            agent_vehicles_factor=1.0,
            normalise_costs=True,
            route_filename=None
    ):
        self.__road_network = Network(net_name, routes_per_od, alt_route_file_name=route_filename)
        self.__road_network.reset_graph()

        # -- Env properties
        self.__agent_vehicles_factor = agent_vehicles_factor
        self.__normalize_costs = normalise_costs

        self.__avg_travel_time = 0
        self.__normalised_avg_travel_time = 0

        # sum of routes' costs through time (used to compute the averages)
        self.routes_costs_sum = {od: [0.0 for _ in range(self.__road_network.get_route_set_size(od))] for od in
                                 self.od_pairs}
        self.routes_costs_min = {od: 0.0 for od in self.od_pairs}

        self.__flow_distribution = self.__road_network.get_empty_solution()
        self.__flow_distribution_w_preferences = self.__road_network.get_empty_solution()

        self.__drivers = {}
        for od in self.od_pairs:
            n_agents = int(
                Decimal(str(self.__road_network.get_OD_flow(od))) / Decimal(str(float(self.__agent_vehicles_factor))))
            self.__drivers.update({
                f'driver_{od}_{i}': EnvDriverAgent(
                    d_id=f'driver_{od}_{i}',
                    flow=self.__agent_vehicles_factor,
                    od_pair=od,
                    preference_money_over_time=0.5  # agent's preference
                )
                for i in range(n_agents)
            })

        # -- Agents
        self.agents = list(self.__drivers.keys())
        self.possible_agents = self.agents[:]

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

        self.__iteration = 0

    # -- Road Network properties
    # -----------------------------
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
    def road_network_flow_distribution(self):
        return self.__flow_distribution

    def get_free_flow_travel_times(self, od: str):
        routes: list = self.__road_network.get_routes(od)
        free_flow_travel_times = [r.get_free_flow_travel_time(self.__normalize_costs) for r in routes]
        return free_flow_travel_times

    def __update_routes_costs_stats(self):
        for od in self.od_pairs:
            for r in range(int(self.__road_network.get_route_set_size(od))):
                cc = self.__road_network.get_route(od, r).get_cost(True)
                # if self.__tolling:
                #     cc = 2 * cc - self.__road_network.get_route(od, r).get_free_flow_travel_time(self.__normalize_costs)
                self.routes_costs_sum[od][r] += cc
            self.routes_costs_min[od] = min(self.routes_costs_sum[od]) / (self.__iteration + 1)

    # -- Environment
    # -----------------
    def step(self, actions):
        """

        :param actions: Dictionary mapping from driver_id to action
        :return:
            obs_n: None, due to the problem being stateless
            reward_n: Travel time of the agent after taking action
            terminal_n: True, due to the problem being stateless
            truncated_n: False, due to the problem being stateless
            info_n: Info on the action taken (free flow travel time of route).

        """
        obs_n = {}
        reward_n = {}
        terminal_n = {}
        truncated_n = {}
        info_n = {}

        self.__flow_distribution = self.__road_network.get_empty_solution()
        self.__flow_distribution_w_preferences = self.__road_network.get_empty_solution()

        # Evaluate solution based on routes taken and flow of drivers
        for d_id, route_id in actions.items():
            try:
                self.__drivers[d_id].set_current_route(route_id)
            except KeyError:
                print(f'Driver {d_id} does not exist in the environment')
                continue
            od_order = self.__road_network.get_OD_order(self.get_driver_od_pair(d_id))
            self.__flow_distribution[od_order][route_id] += self.get_driver_flow(d_id)

        self.__avg_travel_time, self.__normalised_avg_travel_time = self.__road_network.evaluate_assignment(self.__flow_distribution, self.__flow_distribution_w_preferences)

        # Update the sum of routes' costs (used to compute the averages)
        self.__update_routes_costs_stats()

        for d_id in actions.keys():
            obs_n[d_id] = None
            reward_n[d_id] = self.__get_reward(d_id)
            terminal_n[d_id] = True
            truncated_n[d_id] = False
            info_n[d_id] = self.__get_info(d_id)

        # As a single state environment, we:
        # - empty the agents set from the environment
        # - return 'True' for the terminal variable
        self.agents = []

        self.__iteration += 1
        return obs_n, reward_n, terminal_n, truncated_n, info_n

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.agents = list(self.__drivers.keys())

        self.__road_network.reset_graph()

        self.__flow_distribution = self.__road_network.get_empty_solution()
        self.__flow_distribution_w_preferences = self.__road_network.get_empty_solution()

        obs_n = {d_id: self.observation_space(d_id) for d_id in self.agents}
        if not return_info:
            return obs_n

        info_n = {d_id: self.__get_info(d_id) for d_id in self.agents}
        return obs_n, info_n

    def seed(self, seed=None):
        super().seed(seed)

    def render(self):
        raise NotImplementedError

    def close(self):
        del self

    def state(self):
        """
        :return: None, due to the environment being a stateless MDP
        """
        return None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, d_id: AgentID) -> None:
        """
        :param d_id: Agent ID
        :return: None, due to the environment being a stateless MDP
        """
        return None

    @functools.lru_cache(maxsize=None)
    def action_space(self, d_id: AgentID) -> Discrete:
        """
        :param d_id: Agent ID
        :return:
        """
        od_pair: str = self.get_driver_od_pair(d_id)
        return Discrete(self.__road_network.get_route_set_size(od_pair))

    def __get_reward(self, d_id: AgentID) -> float:
        od_pair = self.__drivers[d_id].get_od_pair()
        route_id = self.__drivers[d_id].get_current_route()
        return self.__get_travel_time(od_pair, route_id)

    def __get_info(self, d_id: AgentID) -> dict:
        """
        It returns information about the Origin-Destination pair of an agent:
        - The free flow travel time of an agent's possible routes (actions)

        :param d_id:  Agent ID
        :return: dict
        """
        info = {
            "free_flow_travel_times": self.get_free_flow_travel_times(self.get_driver_od_pair(d_id))
        }
        return info

    # -- Environment Properties
    # ----------------------------
    def __get_travel_time(self, od_pair: str, r_id: int):
        """
        :param r_id:  Route ID
        :return: driver's travel time
        """
        route = self.__road_network.get_route(od_pair, r_id)
        cost = route.get_cost(self.__normalize_costs)
        return cost

    # -- Driver Properties
    # -----------------------
    def get_driver_flow(self, d_id: AgentID) -> float:
        return self.__drivers[d_id].get_flow()

    def get_driver_od_pair(self, d_id: AgentID) -> str:
        return self.__drivers[d_id].get_od_pair()
