import timeit

from route_choice_env.route_choice import RouteChoicePZ
from route_choice_env.core import Policy

from route_choice_env.agents.rmq_learning import RMQLearning
from route_choice_env.policy import EpsilonGreedy

from pettingzoo.utils.conversions import AgentID


def get_env():
    return RouteChoicePZ('OW', 8)


def get_policy() -> Policy:
    return EpsilonGreedy(epsilon=1.0, min_epsilon=0.0)


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
        ITERATIONS=1000,
        DECAY=0.995,
        ALPHA=1.0,
        MIN_ALPHA=0.0
):

    # instantiate env
    env = get_env()

    # instantiate global policy
    policy = get_policy()

    # instantiate learning agents as drivers
    drivers: dict[AgentID, RMQLearning] = get_learning_agents(env, policy)

    best = float('inf')
    for _ in range(ITERATIONS):

        # query for action from each agent's policy
        act_n = {d_id: drivers[d_id].choose_action() for d_id in env.agents}

        # update global policy (epsilon)
        policy.update(DECAY)

        # step environment
        obs_n_, reward_n, terminal_n, truncated_n, info_n = env.step(act_n)

        # test for best avg travel time
        if env.avg_travel_time < best:
            best = env.avg_travel_time

        # Update strategy (Q table)
        for d_id in env.agents:
            drivers[d_id].update_strategy(obs_n_[d_id], reward_n[d_id], info_n[d_id], alpha=ALPHA)

        # update global learning rate (alpha)
        if ALPHA > MIN_ALPHA:
            ALPHA = ALPHA * DECAY
        else:
            ALPHA = MIN_ALPHA

        solution = env.road_network_flow_distribution
        env.reset()

    print(f"Last road_network_flow_distribution: {solution}")
    print(f"Last avg_travel_time: {env.avg_travel_time}")
    print(f"Best avg_travel_time: {best}")

    env.close()


if __name__ == "__main__":
    starttime = timeit.default_timer()
    print("Exp start time is :", starttime)
    main()
    print("Exp time difference is :", timeit.default_timer() - starttime)
