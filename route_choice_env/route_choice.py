import numpy as np
from gymnasium.spaces import Discrete

import functools
from decimal import Decimal
from typing import Mapping, Optional

from route_choice_env.core import Driver
from route_choice_env.misc import Distribution
from route_choice_env.problem import Network

from pettingzoo import ParallelEnv
from pettingzoo.utils.conversions import AgentID


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
            agent_vehicles_factor: float = 1.0,
            revenue_redistribution_rate: float = 0.0,
            normalise_costs: bool = True,
            preference_dist_name: str = None,
            route_filename: str = None,
    ):
        self.__road_network = Network(net_name, routes_per_od, alt_route_file_name=route_filename)
        self.__road_network.reset_graph()

        self.__revenue_redistribution_rate = revenue_redistribution_rate

        # -- Env properties
        self.__agent_vehicles_factor = agent_vehicles_factor
        if preference_dist_name is None:
            self.__preference_money_over_time = Distribution(Distribution.DIST_FIXED)
        else:
            self.__preference_money_over_time = Distribution(dist=Distribution.get_dist_id(preference_dist_name), num_of_samples=self.__road_network.get_total_flow())
        self.__normalize_costs = normalise_costs

        self.__avg_travel_time = 0
        self.__normalised_avg_travel_time = 0

        # sum of routes' costs through time (used to compute the averages)
        self.routes_costs_sum = {od: [0.0 for _ in range(self.__road_network.get_route_set_size(od))] for od in self.od_pairs}
        self.routes_costs_min = {od: 0.0 for od in self.od_pairs}

        self.__flow_distribution = self.__road_network.get_empty_solution()
        self.__flow_distribution_w_preferences = self.__road_network.get_empty_solution()

        self.__drivers: Mapping[AgentID, Driver] = {}
        self.__create_drivers()

        # -- Agents
        self.agents = list(self.__drivers.keys())
        self.possible_agents = self.agents[:]

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

        self.__iteration = 0


        # dev
        # ---
        # if revenue_redistribution_rate > 0:
        self.tolls_share_per_od = [0.0 for _ in range(len(self.__road_network.get_OD_pairs()))]
        self.side_payment_per_od = [0.0 for _ in range(len(self.__road_network.get_OD_pairs()))]
        # ---

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

    def __create_drivers(self):
        for od in self.od_pairs:
            n_agents = int(Decimal(str(self.__road_network.get_OD_flow(od))) / Decimal(str(float(self.__agent_vehicles_factor))))
            self.__drivers.update({
                f'driver_{od}_{i}': Driver(
                    d_id=f'driver_{od}_{i}',
                    od_pair=od,
                    flow=self.__agent_vehicles_factor,
                    preference_money_over_time=self.__preference_money_over_time.sample()  # agent's preference
                )
                for i in range(n_agents)
            })


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
                self.__drivers[d_id].current_route = route_id
            except KeyError:
                print(f'Driver {d_id} does not exist in the environment')
                continue
            od_order = self.__road_network.get_OD_order(self.get_driver_od_pair(d_id))
            self.__flow_distribution[od_order][route_id] += self.get_driver_flow(d_id)


            # dev
            # --- we are assigning the flow to the network (evaluate assignment) only AFTER we calculated the tolls
            # self.tolls_share_per_od[od_order] += self.__get_marginal_cost(self.get_driver_od_pair(d_id), route_id)
            # ---

        self.__avg_travel_time, self.__normalised_avg_travel_time = self.__road_network.evaluate_assignment(self.__flow_distribution, self.__flow_distribution_w_preferences)

        # Update the sum of routes' costs (used to compute the averages)
        self.__update_routes_costs_stats()

        # dev
        # ---
        for d_id, r_id in actions.items():
            od = self.get_driver_od_pair(d_id)
            od_order = self.__road_network.get_OD_order(od)
            preference = self.get_driver_preference_money_over_time(d_id)

            toll = (self.__get_marginal_cost(od, r_id) + self.__get_reward(d_id) * preference) / preference
            self.tolls_share_per_od[od_order] += toll

            # ---
        if self.__revenue_redistribution_rate > 0.0:
            for od in self.road_network.get_OD_pairs():
                od_i: int = self.road_network.get_OD_order(od)
                temp = self.tolls_share_per_od[od_i] * self.__revenue_redistribution_rate
                self.side_payment_per_od[od_i] = temp / self.road_network.get_OD_flow(od)

        # ---

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
            return obs_n, {}

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
        return self.__get_route_travel_time(od_pair, route_id)

    def __get_info(self, d_id: AgentID) -> dict:
        """
        It returns information about the Origin-Destination pair of an agent:
        - The free flow travel time of an agent's possible routes (actions)

        :param d_id:  Agent ID
        :return: dict
        """
        od_pair: str = self.get_driver_od_pair(d_id)
        info = {
            "preference_money_over_time": self.get_driver_preference_money_over_time(d_id),
            "free_flow_travel_times": self.get_free_flow_travel_times(od_pair),
            "marginal_cost": self.__get_marginal_cost(od_pair, self.get_driver_current_route(d_id)),
            "side_payment": self.__get_side_payment_for_od(od_pair)
        }
        return info

    # -- Environment Properties
    # ----------------------------
    def __get_route_travel_time(self, od_pair: str, r_id: int):
        """
        :param r_id:  Route ID
        :return: driver's travel time
        """
        route = self.__road_network.get_route(od_pair, r_id)
        cost = route.get_cost(self.__normalize_costs)
        return cost

    def __get_marginal_cost(self, od_pair: str, r_id: int):
        if r_id not in self.road_network.get_routes_ids(od_pair):
            return 0.0
        return self.__get_route_travel_time(od_pair, r_id) - self.get_free_flow_travel_times(od_pair)[r_id]

    def __get_toll_share_for_od(self, od_pair: str):
        od_i: int = self.road_network.get_OD_order(od_pair)
        return self.tolls_share_per_od[od_i]

    def __get_side_payment_for_od(self, od_pair):
        od_i: int = self.road_network.get_OD_order(od_pair)
        return self.side_payment_per_od[od_i]

    # -- Driver Properties
    # -----------------------
    def get_driver_flow(self, d_id: AgentID) -> float:
        return self.__drivers[d_id].get_flow()

    def get_driver_od_pair(self, d_id: AgentID) -> str:
        return self.__drivers[d_id].get_od_pair()

    def get_driver_current_route(self, d_id: AgentID) -> str:
        return self.__drivers[d_id].get_current_route()

    def get_driver_preference_money_over_time(self, d_id: AgentID) -> float:
        return self.__drivers[d_id].get_preference_money_over_time()
