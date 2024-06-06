import numpy as np
from typing import Dict
from pettingzoo.utils.conversions import AgentID

from route_choice_env.core import Policy
from route_choice_env.route_choice import RouteChoicePZ
from route_choice_env.policy import EpsilonGreedy

from route_choice_env.agents.simple_driver import SimpleDriver
from route_choice_env.agents.rmq_learning import RMQLearning
from route_choice_env.agents.tq_learning import TQLearning
from route_choice_env.agents.gtq_learning import GTQLearning


def get_simple_driver_agents(env: RouteChoicePZ, policy: Policy) -> Dict[AgentID, SimpleDriver]:
    return {
        d_id: SimpleDriver(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            policy=policy
        )
        for d_id in env.agents
    }


def get_rmq_learning_agents(env: RouteChoicePZ, policy: Policy) -> Dict[AgentID, RMQLearning]:
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


def get_tq_learning_agents(env: RouteChoicePZ, policy: Policy) -> Dict[AgentID, TQLearning]:
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


def get_gtq_learning_agents(env: RouteChoicePZ, policy: Policy) -> Dict[AgentID, GTQLearning]:
    obs_n, info_n = env.reset(return_info=True)
    return {
        d_id: GTQLearning(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            extrapolate_costs=False,
            # preference_money_over_time=env.get_driver_preference_money_over_time(d_id),
            policy=policy
        )
        for d_id in env.agents
    }


def simulate(
        alg,
        net,
        k,
        alpha_decay,
        min_alpha,
        epsilon_decay,
        min_epsilon,
        agent_vehicles_factor,
        revenue_redistribution_rate,
        preference_dist_name,
        episodes,
        seed,
        render
):
    if seed:
        np.random.seed(seed)

    route_filename = None
    if net in ['BBraess_1_2100_10_c1_2100', 'BBraess_3_2100_10_c1_900', 'BBraess_5_2100_10_c1_900', 'BBraess_7_2100_10_c1_900']:
        route_filename = f"{net}.TRC.routes"

    # learning rate
    alpha = 1.0

    # initiate environment
    env = RouteChoicePZ(
        net,
        k,
        agent_vehicles_factor,
        revenue_redistribution_rate=revenue_redistribution_rate,
        preference_dist_name=preference_dist_name,
        route_filename=route_filename
        )

    # instantiate global policy
    epsilon = 1.0

    policy = EpsilonGreedy(epsilon, min_epsilon)

    # instantiate learning agents as drivers
    if alg == 'RMQLearning':
        drivers = get_rmq_learning_agents(env, policy)
    elif alg == 'TQLearning':
        drivers = get_tq_learning_agents(env, policy)
    elif alg == 'GTQLearning':
        drivers = get_gtq_learning_agents(env, policy)

    if render:
        env.render()

    best = float('inf')
    for _ in range(episodes):

        # query for action from each agent's policy
        act_n = {d_id: drivers[d_id].choose_action() for d_id in env.agents}

        # update global policy
        policy.update(epsilon_decay)

        # step environment
        obs_n, reward_n, terminal_n, truncated_n, info_n = env.step(act_n)

        if render:
            env.render()

        # test for best avg travel time
        if env.avg_travel_time < best:
            best = env.avg_travel_time

        # update strategy (Q table)
        for d_id in drivers.keys():
            drivers[d_id].update_strategy(obs_n[d_id], reward_n[d_id], info_n[d_id], alpha=alpha)

        # update global learning rate (alpha)
        if alpha > min_alpha:
            alpha = alpha * alpha_decay
        else:
            alpha = min_alpha

        solution = env.road_network_flow_distribution
        env.reset()

    env.close()
