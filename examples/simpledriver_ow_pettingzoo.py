import timeit
from pettingzoo.utils.conversions import AgentID

import route_choice_env.services as services
from route_choice_env.agents.simple_driver import SimpleDriver


def main(
    ITERATIONS = 1000,
):
    # instantiate env
    env = services.get_env()
    # instantiate global policy
    policy = services.get_random_policy()
    # instantiate learning agents as drivers
    drivers: dict[AgentID, "SimpleDriver"] = services.get_simple_driver_agents(env, policy)

    best = float('inf')
    for _ in range(ITERATIONS):

        act_n = {d_id: drivers[d_id].choose_action() for d_id in env.agents}

        obs_n_, reward_n, terminal_n, truncated_n, info_n = env.step(act_n)

        if env.avg_travel_time < best:
            best = env.avg_travel_time

        flow_dist = env.road_network_flow_distribution

        env.reset()

    print(flow_dist)
    print(env.avg_travel_time)
    print(env.routes_costs_sum)
    print(env.routes_costs_min)


if __name__ == '__main__':
    starttime = timeit.default_timer()
    print("Exp start time is :", starttime)
    main()
    print("Exp time difference is :", timeit.default_timer() - starttime)
