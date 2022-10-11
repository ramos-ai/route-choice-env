from route_choice_gym.route_choice import RouteChoice


def get_env() -> RouteChoice:
    return RouteChoice()


def test_env_step():
    env = get_env()

    res = False
    try:
        env.step([])
    except NotImplementedError:
        res = True

    assert res
