import pytest

from route_choice_env.route_choice import RouteChoicePZ


# networks
@pytest.fixture
def network_names():
    return ['Braess_1_4200_10_c1', 'Braess_2_4200_10_c1', 'Braess_3_4200_10_c1', 'Braess_4_4200_10_c1', 'Braess_5_4200_10_c1', 'Braess_6_4200_10_c1', 'Braess_7_4200_10_c1', 'BBraess_1_2100_10_c1_2100', 'BBraess_3_2100_10_c1_900', 'BBraess_5_2100_10_c1_900', 'BBraess_7_2100_10_c1_900', 'OW', 'SF']

# envs
@pytest.fixture
def ow_8_env():
    return RouteChoicePZ('OW', 8)


# agents
