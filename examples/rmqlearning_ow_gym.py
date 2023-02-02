
from route_choice_env.problem import Network
from route_choice_env.route_choice import RouteChoice
from route_choice_env.statistics import Statistics

from route_choice_env.core import DriverAgent, Policy

from route_choice_env.agents.rmq_learning import RMQLearning
from route_choice_env.policy import EpsilonGreedy


# experiment variables
EPSILON = 1.0
MIN_EPSILON = 0.0
ALPHA = 1.0
MIN_ALPHA = 0.0

ITERATIONS = 1000
DECAY = 0.99


def get_env() -> RouteChoice:
    road_network = Network(network_name='OW', routes_per_OD=8)
    return RouteChoice(road_network)


def get_policy() -> Policy:
    return EpsilonGreedy(EPSILON, MIN_EPSILON)


def get_drivers(env: RouteChoice, policy: Policy) -> list[DriverAgent]:
    driver_agents = list()
    for od, n in env.n_agents_per_od.items():
        actions = list(range(env.action_space[od].n))
        free_flow_travel_times = env.get_free_flow_travel_times(od)

        driver_agents += [
            RMQLearning(
                od_pair=od,
                actions=actions,
                initial_costs=free_flow_travel_times,
                policy=policy
            )
            for _ in range(n)
        ]
    return driver_agents


def main():
    # instantiate env
    env = get_env()

    # instantiate global policy
    policy = get_policy()

    # create driver agents
    driver_agents = get_drivers(env, policy)

    return

    # set drivers to the environment
    env.set_drivers(driver_agents)

    # sum of routes' costs through time (used to compute the averages)
    routes_costs_sum = {od: [0.0 for _ in range(env.get_route_set_size(od))] for od in env.od_pairs}

    # sum of the average regret per OD pair (used to measure the averages through time)
    # for each OD pair, it stores a tuple [w, x, y, z]
    # - w the average real regret
    # - x the average estimated regret
    # - y the average absolute difference between them
    # - z the relative difference between them
    sum_regrets = {od: [0.0, 0.0, 0.0, 0.0] for od in env.od_pairs}

    statistics = Statistics(env.road_network, env.drivers, ITERATIONS, True, True, True)
    print("\n")

    best = float('inf')
    obs_n, info_n = env.reset()
    for _ in range(ITERATIONS):

        # query for action from each agent's policy
        act_n = [d.choose_action() for d in env.drivers]

        # update global policy (epsilon)
        policy.update(DECAY)

        # step environment
        obs_n_, reward_n, terminal_n, info_n = env.step(act_n)

        # test for best avg travel time
        if env.avg_travel_time < best:
            best = env.avg_travel_time

        # update strategy (q-table)
        for i, d in enumerate(env.drivers):
            d.update_strategy(obs_n_[i], reward_n[i], info_n[i], alpha=ALPHA)

        # update global learning rate (alpha)
        if ALPHA > MIN_ALPHA:
            ALPHA = ALPHA * DECAY
        else:
            ALPHA = MIN_ALPHA

        # -- episode statistics
        # -------------------------
        for i, d in enumerate(env.drivers):
            try:
                d.update_real_regret(env.routes_costs_min(d.get_od_pair()))
            except AttributeError:  # validation in case driver does not calculate real regret
                pass

        gen_real, gen_estimated, gen_diff, gen_relative_diff, sum_regrets = statistics.print_statistics_episode(
            _,
            env.avg_travel_time,
            sum_regrets
        )

    road_flow_distribution = env.solution

    statistics.print_statistics(road_flow_distribution, env.avg_travel_time, best, sum_regrets, routes_costs_sum)

    env.close()


if __name__ == "__main__":
    main()
