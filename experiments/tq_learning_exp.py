import sys

from route_choice_env.core import Policy
from route_choice_env.route_choice import RouteChoicePZ
from route_choice_env.statistics import Statistics

from route_choice_env.agents.tq_learning import TQLearning
from route_choice_env.policy import EpsilonGreedy

from experiment import Experiment


N_EXPERIMENTS = 0


def get_tq_learning_agents(env: RouteChoicePZ, policy: Policy):
    obs_n, info_n = env.reset(return_info=True)
    return {
        d_id: TQLearning(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            extrapolate_costs=False,
            policy=policy
        )
        for d_id in env.agents
    }


class TQLearningExperiment(Experiment):

    def __init__(self,
                _id: int,
                episodes: int,
                net: str,
                k: int,
                alpha_decay: float,
                epsilon_decay: float,
                revenue_redistribution_rate: float,
                preference_dist_name: str,
                rep: int):
        super(TQLearningExperiment, self).__init__(
            _id,
            'TQLearning',
            episodes,
            net,
            k,
            alpha_decay,
            epsilon_decay,
            revenue_redistribution_rate,
            preference_dist_name,
            rep
        )

    def run_experiment(self, r_id: int):
        sys.stdout = open(f'{self.LOGPATH}/results_v{self.LOG_V}_r{r_id}.txt', 'w')

        print('========================================================================')
        print(f' Experiment {self._ID} of ')
        print(f' algorithm={self.ALG}, network={self.NET}, replication={r_id}, K={self.K}, decay={self.DECAY}')
        print('========================================================================\n')

        route_filename = None
        if self.NET in ['BBraess_1_2100_10_c1_2100', 'BBraess_3_2100_10_c1_900', 'BBraess_5_2100_10_c1_900', 'BBraess_7_2100_10_c1_900']:
            route_filename = f"{self.NET}.TRC.routes"

        # initiate environment
        env = RouteChoicePZ(self.NET, self.K, route_filename=route_filename)

        # instantiate global policy
        policy = EpsilonGreedy(self.EPSILON, self.MIN_EPSILON)

        # instantiate learning agents as drivers
        drivers = get_tq_learning_agents(env, policy)

        # learning rate
        alpha = self.ALPHA

        # sum of the average regret per OD pair (used to measure the averages through time)
        # for each OD pair, it stores a tuple [w, x, y, z]
        # - w the average real regret
        # - x the average estimated regret
        # - y the average absolute difference between them
        # - z the relative difference between them
        sum_regrets = {od: [0.0, 0.0, 0.0, 0.0] for od in env.od_pairs}

        statistics = Statistics(env, drivers, self.ITERATIONS, True, True, True)

        best = float('inf')
        for _ in range(self.ITERATIONS):

            # query for action from each agent's policy
            act_n = {d_id: drivers[d_id].choose_action() for d_id in env.agents}

            # update global policy
            policy.update(self.EPSILON_DECAY)

            # step environment
            obs_n, reward_n, terminal_n, truncated_n, info_n = env.step(act_n)

            # test for best avg travel time
            if env.avg_travel_time < best:
                best = env.avg_travel_time

            # update strategy (Q table)
            for d_id in drivers.keys():
                drivers[d_id].update_strategy(obs_n[d_id], reward_n[d_id], info_n[d_id], alpha=alpha)

            # update global learning rate (alpha)
            if alpha > self.MIN_ALPHA:
                alpha = alpha * self.ALPHA_DECAY
            else:
                alpha = self.MIN_ALPHA

            # -- episode statistics
            # -------------------------
            for d_id, d in drivers.items():
                try:
                    d.update_real_regret(env.routes_costs_min[env.get_driver_od_pair(d_id)])
                except AttributeError:  # validation in case driver does not calculate real regret
                    pass

            gen_real, gen_estimated, gen_diff, gen_relative_diff, sum_regrets = statistics.print_statistics_episode(
                _,
                env.avg_travel_time,
                sum_regrets
            )

            solution = env.road_network_flow_distribution
            env.reset()

        statistics.print_statistics(solution, env.avg_travel_time, best, sum_regrets, env.routes_costs_sum)

        # env.close()

        sys.stdout = sys.__stdout__

        return [env.avg_travel_time, gen_real, gen_estimated, gen_diff, gen_relative_diff]
