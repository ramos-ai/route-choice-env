from typing import List

from route_choice_gym.core import DriverAgent
from route_choice_gym.problem import ProblemInstance


class Statistics(object):

    def __init__(self, problem_instance: ProblemInstance, drivers: List[DriverAgent], iterations, stat_regret_diff,
                 stat_all, print_od_pairs_every_episode: bool):

        self.__problem_instance = problem_instance  # problem instance
        self.__drivers = drivers  # set of drivers

        # parameters of the problem instance
        self.__iterations = iterations
        self.__stat_regret_diff = stat_regret_diff
        self.__stat_all = stat_all
        self.__print_od_pairs_every_episode = print_od_pairs_every_episode

    # -------------------------------------------------------------------

    def print_statistics(self, S, v, best, sum_regrets, routes_costs_sum):

        # print the average regrets of each OD pair along the iterations
        print('\nAverage regrets over all timesteps (real, estimated, absolute difference, relative difference) '
              'per OD pair:')
        for od in self.__problem_instance.get_OD_pairs():
            print(f'\t{od}\t{sum_regrets[od][0] / self.__iterations}\t{sum_regrets[od][1] / self.__iterations}'
                  f'\t{sum_regrets[od][2] / self.__iterations}\t{sum_regrets[od][3] / self.__iterations}')

        # print the average cost of each route of each OD pair along iterations
        print('\nAverage cost of routes:')
        for od in self.__problem_instance.get_OD_pairs():
            print(od)
            for r in range(int(self.__problem_instance.get_route_set_size(od))):
                routes_costs_sum[od][r] /= self.__iterations
                print(f'\t{r}\t{routes_costs_sum[od][r]}')

        print(f'\nLast solution {S} = {v}')
        print(f'Best value found was of {best}')

        # print the average strategy (for each OD pair)
        print('\nAverage strategy per OD pair:')
        for od in self.__problem_instance.get_OD_pairs():
            strategies = {r: 0.0 for r in range(len(self.__problem_instance.get_routes(od)))}
            for d in self.__drivers:
                if d.get_od_pair() == od:
                    S = d.get_strategy()
                    for s in S:
                        strategies[s] += S[s]
            for s in strategies:
                strategies[s] = round(strategies[s] / self.__problem_instance.get_OD_flow(od), 3)
            print(f'\t{od}\t{strategies}')

        print('\nAverage expected cost of drivers per OD pair')
        expected_cost_sum = {od: 0.0 for od in self.__problem_instance.get_OD_pairs()}
        for d in self.__drivers:
            _sum = 0.0
            for r in d.get_strategy():
                _sum += d.get_strategy()[r] * routes_costs_sum[d.get_od_pair()][r]
            expected_cost_sum[d.get_od_pair()] += _sum
        total = 0.0
        for od in self.__problem_instance.get_OD_pairs():
            total += expected_cost_sum[od]
            print(f'{od}\t{expected_cost_sum[od] / self.__problem_instance.get_OD_flow(od)}')
        print(f'Average: {total / self.__problem_instance.get_total_flow()}')

    def print_statistics_episode(self, iteration, v, sum_regrets):

        # store the SUM of regrets over all drivers in the CURRENT timestep
        # for each od [w, x, y, z], where w and x represent the real and estimated
        # regrets, and y and z represent absolute and relative difference between
        # the estimated and real regrets
        regrets = {od: [0.0, 0.0, 0.0, 0.0] for od in self.__problem_instance.get_OD_pairs()}

        gen_real = 0.0
        gen_estimated = 0.0
        gen_diff = 0.0
        gen_relative_diff = 0.0

        # compute the drivers' regret on CURRENT iteration
        for d in self.__drivers:
            # get the regrets
            real = d.get_real_regret()
            estimated = d.get_estimated_regret()

            # store in the appropriate space
            regrets[d.get_od_pair()][0] += real
            regrets[d.get_od_pair()][1] += estimated

            gen_real += real
            gen_estimated += estimated

        if self.__stat_regret_diff:
            for d in self.__drivers:

                # compute the regrets
                real = d.get_real_regret()
                estimated = d.get_estimated_regret()
                diff = abs(estimated - real)
                fxy = max(abs(estimated), abs(real))
                try:
                    relative_diff = (diff / fxy)  # https://en.wikipedia.org/wiki/Relative_change_and_difference
                except ZeroDivisionError:
                    relative_diff = 0.0

                # store in the appropriate space
                regrets[d.get_od_pair()][2] += diff
                regrets[d.get_od_pair()][3] += relative_diff

                gen_diff += diff
                gen_relative_diff += relative_diff

        # calculate the total averages
        gen_real /= self.__problem_instance.get_total_flow()
        gen_estimated /= self.__problem_instance.get_total_flow()
        if self.__stat_regret_diff:
            gen_diff /= self.__problem_instance.get_total_flow()
            gen_relative_diff /= self.__problem_instance.get_total_flow()

        str_print = f'%d\t%f\t%f\t%f' % (iteration, v, gen_real, gen_estimated)
        if self.__stat_regret_diff:
            str_print = '%s\t%f\t%f' % (str_print, gen_diff, gen_relative_diff)

        # calculate the average regrets (real, estimated, absolute difference
        # and relative difference) and then store and plot them (ALL iterations)
        to_print = []
        for od in self.__problem_instance.get_OD_pairs():

            # calculate the averages
            real = regrets[od][0] / self.__problem_instance.get_OD_flow(od)
            estimated = regrets[od][1] / self.__problem_instance.get_OD_flow(od)

            # store (over all timestamps)
            sum_regrets[od][0] += real
            sum_regrets[od][1] += estimated

            # store important information from current iteration
            to_print.append([real, estimated])

        if self.__stat_regret_diff:
            for iod, od in enumerate(self.__problem_instance.get_OD_pairs()):

                # compute the averages
                diff = regrets[od][2] / self.__problem_instance.get_OD_flow(od)
                relative_diff = regrets[od][3] / self.__problem_instance.get_OD_flow(od)

                # store (over all timestamps)
                sum_regrets[od][2] += diff
                sum_regrets[od][3] += relative_diff

                # store important information from current iteration
                to_print[iod].append(diff)
                to_print[iod].append(relative_diff)

        # print important information from current iteration
        if self.__stat_all:
            str_add = ''
            if self.__print_od_pairs_every_episode:
                str_add = '%s' % ('\t'.join(map(str, [item for sublist in to_print for item in sublist])))
            print(f'{str_print}\t{str_add}')

        return gen_real, gen_estimated, gen_diff, gen_relative_diff, sum_regrets
