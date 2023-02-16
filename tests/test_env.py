from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test  # noqa: E402

from route_choice_env.route_choice import RouteChoice, RouteChoicePZ


def test_gym_env():
    # TODO
    #   env = RouteChoice()
    #   check_env(env, skip_render_check=True)
    raise NotImplementedError


def test_pz_env():
    env = RouteChoicePZ('OW', 8)
    parallel_api_test(env, num_cycles=1000)
