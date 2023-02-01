from route_choice_env.route_choice import RouteChoicePZ

from route_choice_env.agents.rmq_learning import RMQLearning
from route_choice_env.policy import EpsilonGreedy


env = RouteChoicePZ('OW', 8)

obs_n, info_n = env.reset(return_info=True)

policy = EpsilonGreedy(epsilon=1.0, min_epsilon=0.0)
drivers = {
    d_id: RMQLearning(
        d_id=d_id,
        actions=list(range(env.action_space(d_id).n)),
        initial_costs=info_n[i]['free_flow_travel_times'],
        extrapolate_costs=True,
        policy=policy
    )
    for i, d_id in enumerate(env.agents)
}

DECAY = 0.955
ALPHA = 1.0
MIN_ALPHA = 0.0

best = float('inf')
print("\n")
for _ in range(1000):

    # query for action from each agent's policy
    act_n = {d_id: drivers[d_id].choose_action() for d_id in drivers.keys()}

    # update global policy
    policy.update(DECAY)

    # step environment
    obs_n_, reward_n, terminal_n, info_n = env.step(act_n)

    v = env.avg_travel_time
    if v < best:
        best = v

    # Update strategy (Q table)
    for d_id in drivers.keys():
        drivers[d_id].update_strategy(obs_n_[d_id], reward_n[d_id], info_n[d_id], alpha=ALPHA)

    # Update alpha
    if ALPHA > MIN_ALPHA:
        ALPHA = ALPHA * DECAY
    else:
        ALPHA = MIN_ALPHA

    # -- episode statistics
    # -------------------------
    # for i, d in enumerate(env.agents):
    #     try:
    #         d.update_real_regret(env.get_routes_costs_min(d.get_od_pair()))
    #     except AttributeError:  # validation in case driver does not calculate real regret
    #         pass
    # gen_real, gen_estimated, gen_diff, gen_relative_diff, sum_regrets = statistics.print_statistics_episode(_, v, sum_regrets)

    print(env.road_network_flow_distribution)

env.close()
