import sys
import random
from typing import Optional, Callable

from route_choice_gym.core import Agent
from route_choice_gym.route_choice import RouteChoice
from route_choice_gym.problem import ProblemInstance


def random_policy(actions_range):
    return random.choice(actions_range)


# Simple agent class for testing the environment
class SimpleDriver(Agent):

    def __init__(self, od_pair, actions, policy_callback=Optional[Callable]):
        super(SimpleDriver, self).__init__()

        self.__OD_pair = od_pair
        self.__last_action = 0

        self.__flow = 1.0
        self.__time_flexibility = 0.5

        # self.__strategy = {a: 0.0 for a in actions}
        self.__actions = actions
        self.policy_callback: Callable = policy_callback

    def choose_action(self, obs):
        """
        :param obs: observation of the agent
        :return: returns an action
        """
        if self.policy_callback is not None:
            self.__last_action = self.policy_callback(self.__actions)
        else:
            self.__last_action = 0
        return self.__last_action

    def update_policy(self, obs_, reward):
        raise NotImplementedError

    def get_OD_pair(self):
        return self.__OD_pair

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
    # initiate environment
    env = create_env()
    n_agents_per_od = env.n_of_agents_per_od

    print(f"Number of agents: {n_agents_per_od}")
    print(f"Action space: {env.action_space}")
    # print(f"Observation space: {env.observation_space}")

    # create agents
    D = []
    for od, n in n_agents_per_od.items():

        actions = range(env.action_space[od].n)
        print(f'Creating {n} drivers for route {od} with the set of actions {actions}')
        for _ in range(n):
            D.append(SimpleDriver(od, actions, policy_callback=random_policy))

    # assign drivers to environment
    env.drivers = D
    print(f'Created {len(env.drivers)} total drivers')

    print("\n")
    obs_n = env.reset()
    for _ in range(5):

        # query for action from each agent's policy
        act_n = []
        for i, d in enumerate(env.drivers):
            act_n.append(d.choose_action( obs_n[i] ))

        print("\n")
        print("-- Debugging ...")
        print(f"act_n: {act_n}")

        # step environment
        obs_n_, reward_n, terminal_n = env.step(act_n)

        print(f"obs_n_: {obs_n_}")
        print(f"reward_n: {reward_n}")

        # for i, d in enumerate(env.drivers):
        #     d.update_policy(obs_n_[i], reward_n[i])

    env.close()


if __name__ == '__main__':
    main()
