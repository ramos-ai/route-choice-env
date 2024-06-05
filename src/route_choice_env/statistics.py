from pathlib import Path
from typing import Dict, Union

import pandas as pd

from route_choice_env.route_choice import RouteChoicePZ

from route_choice_env.agents.rmq_learning import RMQLearning
from route_choice_env.agents.tq_learning import TQLearning


class Statistics(object):
    """
        Class for calculating statistics for experiments, both on every episode or after the experiment is completed.

        It supports the following metrics:
        - Average travel time (avg_tt)
        - Real regret ()
    """

    def __init__(self,
                 env: RouteChoicePZ,
                 driver_agents: Dict[str, Union[RMQLearning, TQLearning]],
                 iterations,
                 stat_regret_diff,
                 stat_all,
                 print_od_pairs_every_episode: bool
    ):
        self.__env = env
        self.__road_network = env.road_network
        self.__driver_agents = driver_agents  # set of drivers

        # parameters of the problem instance
        self.__iterations = iterations
        self.__stat_regret_diff = stat_regret_diff
        self.__stat_all = stat_all
        self.__print_od_pairs_every_episode = print_od_pairs_every_episode

        self.__cols_episode = ['i', 'avg_tt', 'real_reg', 'est_reg']
        if self.__stat_regret_diff:
            self.__cols_episode.extend(['abs_diff', 'rel_diff'])

        if self.__print_od_pairs_every_episode:
            for od in self.__road_network.get_OD_pairs():
                self.__cols_episode.extend([f'{od}_avg_real_reg', f'{od}_avg_est_reg'])
                if self.__stat_regret_diff:
                    self.__cols_episode.extend([f'{od}_avg_abs_diff', f'{od}_avg_rel_diff'])  # regret diff

        self.__df_episode_stats = pd.DataFrame(columns=self.__cols_episode)

        print('\t'.join(map(str, [col for col in self.__cols_episode])))

    # -------------------------------------------------------------------

    def print_statistics(self, S, v, best, sum_regrets, routes_costs_sum):

        # print the average regrets of each OD pair along the iterations
        print('\nAverage regrets over all timesteps (real, estimated, absolute difference, relative difference) '
              'per OD pair:')
        for od in self.__road_network.get_OD_pairs():
            print(f'\t{od}\t{sum_regrets[od][0] / self.__iterations}\t{sum_regrets[od][1] / self.__iterations}'
                  f'\t{sum_regrets[od][2] / self.__iterations}\t{sum_regrets[od][3] / self.__iterations}')

        # print the average cost of each route of each OD pair along iterations
        print('\nAverage cost of routes:')
        for od in self.__road_network.get_OD_pairs():
            print(od)
            for r in range(int(self.__road_network.get_route_set_size(od))):
                routes_costs_sum[od][r] /= self.__iterations
                print(f'\t{r}\t{routes_costs_sum[od][r]}')

        print(f'\nLast solution {S} = {v}')
        print(f'Best value found was of {best}')

        # print the average strategy (for each OD pair)
        print('\nAverage strategy per OD pair:')
        for od in self.__road_network.get_OD_pairs():
            strategies = {r: 0.0 for r in range(len(self.__road_network.get_routes(od)))}
            for d_id, d in self.__driver_agents.items():
                if self.__env.get_driver_od_pair(d_id) == od:
                    S = d.get_strategy()
                    for s in S:
                        strategies[s] += S[s]
            for s in strategies:
                strategies[s] = round(strategies[s] / self.__road_network.get_OD_flow(od), 3)
            print(f'\t{od}\t{strategies}')

        print('\nAverage expected cost of drivers per OD pair')
        expected_cost_sum = {od: 0.0 for od in self.__road_network.get_OD_pairs()}
        for d_id, d in self.__driver_agents.items():
            _od_pair = self.__env.get_driver_od_pair(d_id)
            _sum = 0.0
            for r in d.get_strategy():
                _sum += d.get_strategy()[r] * routes_costs_sum[_od_pair][r]
            expected_cost_sum[_od_pair] += _sum
        total = 0.0
        for od in self.__road_network.get_OD_pairs():
            total += expected_cost_sum[od]
            print(f'{od}\t{expected_cost_sum[od] / self.__road_network.get_OD_flow(od)}')
        print(f'Average: {total / self.__road_network.get_total_flow()}')

    def print_statistics_episode(self, iteration, avg_travel_time, sum_regrets):

        # store the SUM of regrets over all drivers in the CURRENT timestep
        # for each od [w, x, y, z], where w and x represent the real and estimated
        # regrets, and y and z represent absolute and relative difference between
        # the estimated and real regrets
        regrets = {od: [0.0, 0.0, 0.0, 0.0] for od in self.__road_network.get_OD_pairs()}

        gen_real = 0.0
        gen_estimated = 0.0
        gen_diff = 0.0
        gen_relative_diff = 0.0

        # compute the drivers' regret on CURRENT iteration
        for d_id, d in self.__driver_agents.items():
            _od_pair = self.__env.get_driver_od_pair(d_id)

            # get the regrets
            real = d.get_real_regret()
            estimated = d.get_estimated_regret()

            # store in the appropriate space
            regrets[_od_pair][0] += real
            regrets[_od_pair][1] += estimated

            gen_real += real
            gen_estimated += estimated

            if self.__stat_regret_diff:
                # compute the regrets
                diff = abs(estimated - real)
                fxy = max(abs(estimated), abs(real))
                try:
                    relative_diff = (diff / fxy)  # https://en.wikipedia.org/wiki/Relative_change_and_difference
                except ZeroDivisionError:
                    relative_diff = 0.0

                # store in the appropriate space
                regrets[_od_pair][2] += diff
                regrets[_od_pair][3] += relative_diff

                gen_diff += diff
                gen_relative_diff += relative_diff

        # calculate the total averages
        gen_real /= self.__road_network.get_total_flow()
        gen_estimated /= self.__road_network.get_total_flow()

        # store values for episode statistics
        episode_stats = [iteration, avg_travel_time, gen_real, gen_estimated]

        if self.__stat_regret_diff:
            gen_diff /= self.__road_network.get_total_flow()
            gen_relative_diff /= self.__road_network.get_total_flow()

            episode_stats.extend([gen_diff, gen_relative_diff])

        # calculate the average regrets (real, estimated, absolute difference and relative difference)
        # and then store and plot them (ALL iterations)
        episode_stats_per_od = {}
        for iod, od in enumerate(self.__road_network.get_OD_pairs()):

            # calculate the averages
            real = regrets[od][0] / self.__road_network.get_OD_flow(od)
            estimated = regrets[od][1] / self.__road_network.get_OD_flow(od)

            # store (over all timestamps)
            sum_regrets[od][0] += real
            sum_regrets[od][1] += estimated

            # store important information from current iteration
            episode_stats_per_od[od] = [real, estimated]

            if self.__stat_regret_diff:
                # compute the averages
                diff = regrets[od][2] / self.__road_network.get_OD_flow(od)
                relative_diff = regrets[od][3] / self.__road_network.get_OD_flow(od)

                # store (over all timestamps)
                sum_regrets[od][2] += diff
                sum_regrets[od][3] += relative_diff

                # store important information from current iteration
                episode_stats_per_od[od].extend([diff, relative_diff])

        # print important information from current iteration
        if self.__stat_all:
            if self.__print_od_pairs_every_episode:
                [episode_stats.extend(stats) for od, stats in episode_stats_per_od.items()]

            self.__df_episode_stats = pd.concat([
                self.__df_episode_stats,
                pd.DataFrame([episode_stats], columns=self.__cols_episode)
            ])

            print('\t'.join(map(str, [stats for stats in episode_stats])))

        return gen_real, gen_estimated, gen_diff, gen_relative_diff, sum_regrets

    def save_episode_stats_csv(self, filename):
        filepath = str(Path(__file__).parent.parent.absolute()) + f"/analytics/data/{filename}.csv"
        self.__df_episode_stats.to_csv(filepath, sep=';')
