"""
    Main process for running experiments on the route choice environment.

    Make sure to have your experiment implemented and your agent configured on the valid_experiments_alg object.
"""
import json
import timeit
import argparse
import numpy as np
from pathlib import Path

from experiment import Experiment

from rmq_learning_exp import RMQLearningExperiment
from tq_learning_exp import TQLearningExperiment
from gtq_learning_exp import GTQLearningExperiment


class InvalidAlgorithm(Exception):
    pass


def run_experiment(Experiment: Experiment, workers: int):
    with open(str(Path(__file__).parent.absolute()) + "/experiments_config.json", 'r') as file:
        raw_experiments = json.load(file)

    _id = 1
    experiments = []
    for _exp in raw_experiments:
        for net in _exp['net']:
            for k in _exp['k']:
                for alpha_decay in _exp['alpha_decays']:
                    for epsilon_decay in _exp['epsilon_decays']:
                        for revenue_redistribution_rate in _exp['revenue_redistribution_rate']:
                            for preference_dist_name in _exp['preference_dist_name']:
                                experiments.append(
                                    Experiment(
                                        _id,
                                        _exp['episodes'],
                                        net,
                                        k,
                                        alpha_decay,
                                        epsilon_decay,
                                        revenue_redistribution_rate,
                                        preference_dist_name,
                                        _exp['rep']
                                    )
                                )
                    _id += 1

    print(f'Running {len(experiments)} experiments...')

    for exp in experiments:
        print(exp)

        if workers > 1:
            exp.run_multiprocess(workers)
        else:
            exp.run()


def main():
    valid_experiments_alg = {
        'RMQLearning': RMQLearningExperiment,
        'TQLearning': TQLearningExperiment,
        'GTQLearning': GTQLearningExperiment,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        EXP: Experiment = valid_experiments_alg[args.alg]
        workers: int = int(args.workers)
        seed: int = int(args.seed) if args.seed is not None else None
    except KeyError:
        raise InvalidAlgorithm(f"Invalid algorithm provided: {args.alg}. "
                               f"\nChoose one from the list: {valid_experiments_alg.keys()}")
    except ValueError:
        raise ValueError(f"Invalid number of workers provided: {args.workers}. "
                         "\nPlease provide a valid integer number of workers.")

    if seed is not None:
        np.random.seed(seed)

    starttime = timeit.default_timer()
    print("The start time is :", starttime)
    run_experiment(EXP, workers)
    print("The time difference is :", timeit.default_timer() - starttime)


if __name__ == '__main__':
    main()
