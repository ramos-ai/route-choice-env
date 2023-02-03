import timeit

from route_choice_env.route_choice import RouteChoice
from route_choice_env.problem import Network

from route_choice_env.agents.rmq_learning import RMQLearningDriver
from route_choice_env.policy import EpsilonGreedy

from route_choice_env.core import DriverAgent, Policy


def get_env() -> RouteChoice:
    road_network = Network(network_name='OW', routes_per_OD=8)
    return RouteChoice(road_network)


def get_policy() -> Policy:
    return EpsilonGreedy(epsilon=1.0, min_epsilon=0.0)


def get_drivers(env: RouteChoice, policy: Policy) -> list[DriverAgent]:
    driver_agents = list()
    for od, n in env.n_agents_per_od.items():
        actions = list(range(env.action_space[od].n))
        free_flow_travel_times = env.get_free_flow_travel_times(od)

        driver_agents += [
            RMQLearningDriver(
                od_pair=od,
                actions=actions,
                initial_costs=free_flow_travel_times,
                policy=policy
            )
            for _ in range(n)
        ]
    return driver_agents


def main(
        ITERATIONS=1000,
        DECAY=0.995,
        ALPHA=1.0,
        MIN_ALPHA=0.0
):

    # initiate environment
    env = get_env()

    # instantiate global policy
    policy = get_policy()

    # create driver agents
    driver_agents = get_drivers(env, policy)

    # assign drivers to environment
    env.set_drivers(driver_agents)

    best = float('inf')
    obs_n, info_n = env.reset()
    for _ in range(ITERATIONS):

        # query for action from each agent's policy
        act_n = [d.choose_action() for d in env.drivers]

        # update global policy
        policy.update(DECAY)

        # step environment
        obs_n_, reward_n, terminal_n, info_n = env.step(act_n)

        # test for best avg travel time
        if env.avg_travel_time < best:
            best = env.avg_travel_time

        # update strategy (Q table)
        for i, d in enumerate(env.drivers):
            d.update_strategy(obs_n_[i], reward_n[i], info_n[i], alpha=ALPHA)

        # update global learning rate (alpha)
        if ALPHA > MIN_ALPHA:
            ALPHA = ALPHA * DECAY
        else:
            ALPHA = MIN_ALPHA

    print(f"Last road_network_flow_distribution: {env.solution}")
    print(f"Last avg_travel_time: {env.avg_travel_time}")
    print(f"Best avg_travel_time: {best}")

    env.close()


if __name__ == "__main__":
    starttime = timeit.default_timer()
    print("The start time is :", starttime)
    main()
    print("The time difference is :", timeit.default_timer() - starttime)
