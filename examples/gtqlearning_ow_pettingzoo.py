import timeit
from pettingzoo.utils.conversions import AgentID

from route_choice_env.route_choice import RouteChoicePZ
from route_choice_env.core import Policy

from route_choice_env.agents.gtq_learning import GTQLearning
from route_choice_env.policy import EpsilonGreedy


def get_env():
    return RouteChoicePZ('OW', 8)


def get_policy() -> Policy:
    return EpsilonGreedy(epsilon=1.0, min_epsilon=0.0)


def get_agents(env: RouteChoicePZ, policy: Policy):
    obs_n, info_n = env.reset(return_info=True)
    return {
        d_id: GTQLearning(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
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

    env = get_env()

    policy = get_policy()

    drivers: dict[AgentID, GTQLearning] = get_agents(env, policy)

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
        if _ == 0:
            print(solution)
            print(env.avg_travel_time)

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
