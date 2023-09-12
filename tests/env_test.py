from pettingzoo.test import parallel_api_test


def test_pz_env(ow_8_env):
    parallel_api_test(ow_8_env, num_cycles=1000)
