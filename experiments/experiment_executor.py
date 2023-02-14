"""
    Main process for running experiments on the route choice environment.

    Make sure to have your experiment implemented and your agent configured on the valid_experiments_alg object.
"""
import json
import timeit
import argparse
from pathlib import Path

from experiments.experiment import Experiment

from experiments.rmq_learning_exp import RMQLearningExperiment
from experiments.tq_learning_exp import TQLearningExperiment


class InvalidAlgorithm(Exception):
    pass


def run_experiment(Experiment, workers):
    with open(str(Path(__file__).parent.absolute()) + "/experiments_config.json", 'r') as file:
        raw_experiments = json.load(file)

    _id = 1
    experiments = []
    for _exp in raw_experiments:
        for net in _exp['net']:
            for k in _exp['k']:
                for decay in _exp['decays']:
                    experiments.append(Experiment(_id, _exp['episodes'], net, k, decay, _exp['rep']))
                    _id += 1

    print(f'Running {len(experiments)} experiments...')

    for exp in experiments:
        if workers > 1:
            exp.run_multiprocess(workers)
        else:
            exp.run()


if __name__ == '__main__':
    valid_experiments_alg = {
        'RMQLearning': RMQLearningExperiment,
        'TQLearning': TQLearningExperiment
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    try:
        EXP: Experiment = valid_experiments_alg[args.alg]
        workers = int(args.workers)
    except KeyError:
        raise InvalidAlgorithm(f"Invalid algorithm provided: {args.alg}. "
                               f"\nChoose one from the list: {valid_experiments_alg.keys()}")

    starttime = timeit.default_timer()
    print("The start time is :", starttime)
    run_experiment(EXP, args.workers)
    print("The time difference is :", timeit.default_timer() - starttime)
