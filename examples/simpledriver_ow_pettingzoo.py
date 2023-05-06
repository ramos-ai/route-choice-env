import timeit
from pettingzoo.utils.conversions import AgentID

from route_choice_env.route_choice import RouteChoicePZ
from route_choice_env.core import Policy

from route_choice_env.agents.simple_driver import SimpleDriver
from route_choice_env.policy import Random


def get_env():
    return RouteChoicePZ('OW', 8)


def get_policy() -> Policy:
    return Random()


def get_agents(env: RouteChoicePZ, policy: Policy):
    return {
        d_id: SimpleDriver(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            policy=policy
        )
        for d_id in env.agents
    }


def main(
    ITERATIONS = 1000,
):

    env = get_env()

    policy = get_policy()

    drivers: dict[AgentID, SimpleDriver] = get_agents(env, policy)

    best = float('inf')
    for _ in range(ITERATIONS):

        act_n = {d_id: drivers[d_id].choose_action() for d_id in env.agents}

        obs_n_, reward_n, terminal_n, truncated_n, info_n = env.step(act_n)

        if env.avg_travel_time < best:
            best = env.avg_travel_time

        flow_dist = env.road_network_flow_distribution
        print(flow_dist)

        env.reset()

    print(flow_dist)
    print(env.avg_travel_time)
    print(env.routes_costs_sum)
    print(env.routes_costs_min)


if __name__ == '__main__':
    main()
