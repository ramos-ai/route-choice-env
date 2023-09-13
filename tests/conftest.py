import sys
from pathlib import Path


def load_modules():
    print(str(Path(__file__).parent.parent))
    sys.path.append(str(Path(__file__).parent.parent))


load_modules()

import pytest
from route_choice_env.route_choice import RouteChoicePZ


# envs
@pytest.fixture
def ow_8_env():
    return RouteChoicePZ('OW', 8)
