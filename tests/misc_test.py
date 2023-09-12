from route_choice_env.misc import Distribution


def test_can_instantiate_distribution():
    _dist_types = Distribution.get_list_of_distributions()

    res = []
    for _type in _dist_types:
        dist = Distribution(dist=_type)
        res.append( isinstance(dist, Distribution) )

        # dist.plot_distribution()
    assert all(res)
