from argparse import ArgumentParser

from route_choice_env.services import simulate


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--alg",
        choices=['RMQLearning', 'TQLearning', 'GTQLearning'],
        required=True,
        )

    parser.add_argument(
        "--net",
        choices=[
            'Anaheim',
            'BBraess_1_2100_10_c1_2100',
            'BBraess_3_2100_10_c1_900',
            'BBraess_5_2100_10_c1_900',
            'BBraess_7_2100_10_c1_900',
            'Braess_1_4200_10_c1',
            'Braess_3_4200_10_c1',
            'Braess_5_4200_10_c1',
            'Braess_7_4200_10_c1',
            'Eastern-Massachusetts',
            'OW',
            'SF',
        ],
        help="Network name",
        required=True,
        )

    parser.add_argument(
        "--k",
        help="Number of routes per origin-destination pair",
        required=True,
        type=int,
        )

    parser.add_argument(
        "--agent_vehicles_factor",
        default=1,
        help="Number of vehicles per agent (default: 1)",
        type=int,
        )

    parser.add_argument(
        "--alpha_decay",
        help="Decay rate for the learning rate",
        required=True,
        type=float,
        )

    parser.add_argument(
        "--min_alpha",
        default=0.0,
        help="Minimum learning rate (default: 0.0)",
        type=float,
        )

    parser.add_argument(
        "--epsilon_decay",
        help="Decay rate for the epsilon-greedy policy",
        required=True,
        type=float,
        )

    parser.add_argument(
        "--min_epsilon",
        default=0.0,
        help="Minimum decay rate for the epsilon-greedy policy (default: 0.0)",
        type=float,
        )

    parser.add_argument(
        "--revenue_redistribution_rate",
        default=0.0,
        help="Rate of revenue redistribution collected from tolls to drivers (default: 0.0)",
        type=float,
        )

    parser.add_argument(
        "--preference_dist_name",
        choices=['DIST_FIXED', 'DIST_UNIFORM', 'DIST_NORMAL', 'DIST_TRUNC_NORMAL'],
        default='DIST_FIXED',
        help="Distribution of driver preferences",
        )

    parser.add_argument(
        "--episodes",
        help="Number of episodes to run",
        required=True,
        type=int,
        )

    parser.add_argument(
        "--seed",
        help="Random seed",
        default=None,
        type=int,
        )

    parser.add_argument(
        "--render",
        help="Render environment",
        action='store_true',
        default=False,
        )

    args = parser.parse_args()

    simulate(
        args.alg,
        args.net,
        args.k,
        args.alpha_decay,
        args.min_alpha,
        args.epsilon_decay,
        args.min_epsilon,
        args.agent_vehicles_factor,
        args.revenue_redistribution_rate,
        args.preference_dist_name,
        args.episodes,
        args.seed,
        args.render,
        )
