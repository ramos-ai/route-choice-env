import timeit
from pettingzoo.utils.conversions import AgentID

import route_choice_env.services as services
from route_choice_env.agents.rmq_learning import RMQLearning


def main(
        ITERATIONS=1000,
        DECAY=0.995,
        ALPHA=1.0,
        MIN_ALPHA=0.0
):
    # instantiate env
    env = services.get_env()
    # instantiate global policy
    policy = services.get_epsilon_greedy_policy()
    # instantiate learning agents as drivers
    drivers: dict[AgentID, RMQLearning] = services.get_rmq_learning_agents(env, policy)

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
