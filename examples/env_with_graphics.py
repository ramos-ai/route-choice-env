import timeit
from argparse import ArgumentParser
from time import time, sleep

from route_choice_env.route_choice import RouteChoicePZ
from route_choice_env.core import Policy

from route_choice_env.agents.rmq_learning import RMQLearning
from route_choice_env.agents.simple_driver import SimpleDriver

from route_choice_env.policy import EpsilonGreedy, Random

from pettingzoo.utils.conversions import AgentID


def get_env(net):
    routes_per_od = {
        'BBraess_7_2100_10_c1_900': 4,
        'Braess_1_4200_10_c1': 3,
        'Braess_7_4200_10_c1': 16,
        'OW': 12,
        'SF': 12,
        'Anaheim': 16,
        'Eastern-Massachusetts': 16,
    }

    return RouteChoicePZ(net, routes_per_od[net])


def get_policy() -> Policy:
    # return Random()
    return EpsilonGreedy(epsilon=1.0, min_epsilon=0.0)


def get_simple_agents(env: RouteChoicePZ, policy: Policy) -> dict[AgentID, SimpleDriver]:
    return {
        d_id: SimpleDriver(
            actions=list(range(env.action_space(d_id).n)),
            d_id=d_id,
            policy=policy
        )
        for d_id in env.agents
    }

def get_learning_agents(env: RouteChoicePZ, policy: Policy) -> dict[AgentID, RMQLearning]:
    obs_n, info_n = env.reset(return_info=True)
    return {
        d_id: RMQLearning(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            initial_costs=info_n[d_id]['free_flow_travel_times'],
            extrapolate_costs=True,
            policy=policy
        )
        for d_id in env.agents
    }


def main(
        NET='OW',
        ITERATIONS=1000,
        DECAY=0.995,
        ALPHA=1.0,
        MIN_ALPHA=0.0
):

    # instantiate env
    env = get_env(NET)
    env.render()

    print('starting simulation...')
    sleep(2)

    # instantiate global policy
    policy = get_policy()

    # instantiate learning agents as drivers
    drivers: dict[AgentID, RMQLearning] = get_simple_agents(env, policy)

    best = float('inf')
    for _ in range(ITERATIONS):

        # query for action from each agent's policy
        act_n = {d_id: drivers[d_id].choose_action() for d_id in env.agents}

        # update global policy (epsilon)
        policy.update()

        # step environment
        obs_n_, reward_n, terminal_n, truncated_n, info_n = env.step(act_n)

        env.render()

        # test for best avg travel time
        if env.avg_travel_time < best:
            best = env.avg_travel_time

        solution = env.road_network_flow_distribution

        env.reset()

    print(f"Last road_network_flow_distribution: {solution}")
    print(f"Last avg_travel_time: {env.avg_travel_time}")
    print(f"Best avg_travel_time: {best}")

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net', type=str, default='OW')
    args = parser.parse_args()

    starttime = timeit.default_timer()
    print("Exp start time is :", starttime)
    main(args.net)
    print("Exp time difference is :", timeit.default_timer() - starttime)
