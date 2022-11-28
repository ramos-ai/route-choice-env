from route_choice_gym.route_choice import RouteChoice
from route_choice_gym.problem import ProblemInstance

from route_choice_gym.agents.rmq_learning import RMQLearning
from route_choice_gym.policy import EpsilonGreedy


def create_env():
    P = ProblemInstance('OW')
    return RouteChoice(P)


def main():
    ITERATIONS = 10000

    ALPHA = 1.0
    ALPHA_DECAY = 0.99
    MIN_ALPHA = 0.0

    EPSILON = 1.0
    EPSILON_DECAY = 0.99
    MIN_EPSILON = 0.0

    # initiate environment
    env = create_env()
    n_agents_per_od = env.n_of_agents_per_od

    print(f"Number of agents: {n_agents_per_od}")
    print(f"Action space: {env.action_space}")
    # print(f"Observation space: {env.observation_space}")

    # create agents
    D = []
    policy = EpsilonGreedy(EPSILON, MIN_EPSILON)
    for od, n in n_agents_per_od.items():

        actions = range(env.action_space[od].n)
        print(f'Creating {n} drivers for route {od} with the set of actions {actions}')

        for _ in range(n):
            D.append(RMQLearning(od, actions, policy=policy))

    # assign drivers to environment
    env.set_drivers(D)
    print(f'Created {len(env.drivers)} total drivers')

    print("\n")
    obs_n = env.reset()
    for _ in range(ITERATIONS):

        # query for action from each agent's policy
        act_n = []
        for i, d in enumerate(env.drivers):
            act_n.append(d.choose_action( obs_n[i] ))

        # Update policy
        policy.update_policy(EPSILON_DECAY)

        # step environment
        obs_n_, reward_n, terminal_n = env.step(act_n)

        print("\n")
        print("-- Debugging ...")
        print(f"act_n: {act_n}")
        print(f"obs_n_: {obs_n_}")
        print(f"reward_n: {reward_n}")

        # Update strategy (Q table)
        for i, d in enumerate(env.drivers):
            d.update_strategy(obs_n_[i], reward_n[i], alpha=ALPHA)

        # Update alpha
        if ALPHA > MIN_ALPHA:
            ALPHA = ALPHA * ALPHA_DECAY
        else:
            ALPHA = MIN_ALPHA

    env.close()


if __name__ == '__main__':
    main()
