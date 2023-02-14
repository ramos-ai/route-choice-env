from pettingzoo.test import parallel_api_test  # noqa: E402

from route_choice_env.route_choice import RouteChoice, RouteChoicePZ


def test_env():
    env = RouteChoicePZ()
    parallel_api_test(env, num_cycles=1000)
