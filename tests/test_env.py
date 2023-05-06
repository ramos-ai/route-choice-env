from pettingzoo.test import parallel_api_test  # noqa: E402

from route_choice_env.route_choice import RouteChoicePZ


def test_pz_env():
    env = RouteChoicePZ('OW', 8)
    parallel_api_test(env, num_cycles=1000)
