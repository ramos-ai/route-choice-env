import sys

from route_choice_env.route_choice import RouteChoice
from route_choice_env.problem import Network
from route_choice_env.statistics import Statistics

from route_choice_env.agents.tq_learning import TQLearning
from route_choice_env.policy import EpsilonGreedy

from experiments.experiment import Experiment


N_EXPERIMENTS = 0


class TQLearningExperiment(Experiment):

    def __init__(self, _id: int, episodes: int, net: str, k: int, decay: float, rep: int):
        super(TQLearningExperiment, self).__init__(
            _id,
            'TQLearning',
            episodes,
            net,
            k,
            decay,
            rep
        )

    def run_experiment(self, r_id: int):
        sys.stdout = open(f'{self.LOGPATH}/results_v{self.LOG_V}_r{r_id}.txt', 'w')

        print('========================================================================')
        print(f' Experiment {self._ID} of ')
        print(f' algorithm={self.ALG}, network={self.NET}, replication={r_id}, K={self.K}, decay={self.DECAY}')
        print('========================================================================\n')

        alt_route_filename = None
        if self.NET in ['BBraess_1_2100_10_c1_2100', 'BBraess_3_2100_10_c1_900', 'BBraess_5_2100_10_c1_900', 'BBraess_7_2100_10_c1_900']:
            alt_route_filename = f"{self.NET}.TRC.routes"

        # create network graph and routes
        road_network = Network(self.NET, self.K, alt_route_filename)

        # initiate environment
        env = RouteChoice(road_network, tolling=True)
        n_agents_per_od = env.n_of_agents_per_od

        # create agents
        driver_agents = []
        policy = EpsilonGreedy(self.EPSILON, self.MIN_EPSILON)
        for od, n in n_agents_per_od.items():

            actions = list(range(env.action_space[od].n))
            for _ in range(n):
                driver_agents.append(TQLearning(od, actions, policy=policy))

        # assign drivers to environment
        env.set_drivers(driver_agents)

        # sum of routes' costs along time (used to compute the averages)
        routes_costs_sum = {od: [0.0 for _ in range(road_network.get_route_set_size(od))] for od in env.od_pairs}

        # sum of the average regret per OD pair (used to measure the averages through time)
        # for each OD pair, it stores a tuple [w, x, y, z], with w the average
        # real regret, x the average estimated regret, y the average absolute
        # difference between them, and z the relative difference between them
        sum_regrets = {od: [0.0, 0.0, 0.0, 0.0] for od in env.od_pairs}

        statistics = Statistics(env.road_network, env.drivers, self.ITERATIONS, True, True, True)

        best = float('inf')

        print("\n")
        obs_n, info_n = env.reset()
        for _ in range(self.ITERATIONS):

            # query for action from each agent's policy
            act_n = []
            for i, d in enumerate(env.drivers):
                act_n.append(d.choose_action())

            # update global policy
            policy.update(self.EPSILON_DECAY)

            # step environment
            obs_n_, reward_n, terminal_n, info_n = env.step(act_n)

            v = env.avg_travel_time
            if v < best:
                best = v

            # Update strategy (Q table)
            for i, d in enumerate(env.drivers):
                d.update_strategy(obs_n_[i], reward_n[i], info_n[i], alpha=self.ALPHA)

            # Update alpha
            if self.ALPHA > self.MIN_ALPHA:
                self.ALPHA = self.ALPHA * self.ALPHA_DECAY
            else:
                self.ALPHA = self.MIN_ALPHA

            # -- episode statistics
            # -------------------------
            for i, d in enumerate(env.drivers):
                try:
                    d.update_real_regret(env.get_routes_costs_min(d.get_od_pair()))
                except AttributeError:  # validation in case driver does not calculate real regret
                    pass

            gen_real, gen_estimated, gen_diff, gen_relative_diff, sum_regrets = statistics.print_statistics_episode(_, v, sum_regrets)

        solution = env.solution

        env.close()

        statistics.print_statistics(solution, v, best, sum_regrets, routes_costs_sum)

        sys.stdout = sys.__stdout__

        return [v, gen_real, gen_estimated, gen_diff, gen_relative_diff]
