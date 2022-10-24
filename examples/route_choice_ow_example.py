import sys
from route_choice_gym.route_choice import RouteChoice
from route_choice_gym.problem import ProblemInstance


# Simple agent class for testing the environment
class SimpleAgent:

    def __init__(self, od_index, actions):
        self.__OD_index = od_index
        self.__strategy = { a: 0.0 for a in actions }
        self.__flow = 1.0
        self.__time_flexibility = 0.5
        self.__last_action = None

    def choose_action(self, obs):
        """
        :param obs: observation of the agent
        :return: returns an action
        """
        self.__last_action = 0
        return 0

    def update_policy(self, obs_, reward):
        # ignore obs
        return 0

    def get_OD_index(self):
        return self.__OD_index

    def get_flow(self):
        return self.__flow

    def get_time_flexibility(self):
        return self.__time_flexibility

    def get_last_action(self):
        return self.__last_action


def create_env():
    P = ProblemInstance('OW')
    return RouteChoice(P)


def main():
    # Initiate environment
    env = create_env()
    n_agents = env.n_of_agents

    # Create agents
    D = []
    for od_index, n in enumerate(n_agents):

        actions = range(env.action_space[od_index].n)
        print(f'Creating {n} drivers for route {od_index} with the set of actions {actions}')
        for _ in range(n):
            D.append(SimpleAgent(od_index, actions))

    env.drivers = D
    print()
    print(f'Created {len(env.drivers)} total drivers')

    obs_n = env.reset()
    for _ in range(1000):

        # query for action from each agent's policy
        act_n = []
        for i, d in enumerate(env.drivers):
            act_n.append(d.choose_action(obs_n[i]))

        # step environment
        obs_n_, reward_n, terminal_n = env.step(act_n)

        # for i, d in enumerate(env.drivers):
        #     D[i].update_policy(obs_n_[i], reward_n[i])

    env.close()


if __name__ == '__main__':
    main()
