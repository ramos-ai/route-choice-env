import sys
import time
import numpy as np

from experiments.tq_learning_exp import TQLearningExperiment


def test_tqlearning_is_consistent():
    expected_results = {
        'Braess_1_4200_10_c1': [4, 0.99, 0.99, (15.467999999991706, 0.055672687074637725, 0.09941309415810554)],
        'Braess_2_4200_10_c1': [8, 0.99, 0.99, (24.965849206343993, 0.038896564625646904, 0.4075917605190364)],
        'Braess_3_4200_10_c1': [8, 0.99, 0.99, (34.845066326528766, 0.026036896258498102, 0.4933477428192948)],
        'Braess_4_4200_10_c1': [12, 0.99, 0.99, (44.79582539682733, 0.018225578231293515, 0.41942431972789)],
        'Braess_5_4200_10_c1': [12, 0.99, 0.99, (54.80873866213191, 0.014663223733936609, 0.3477584618291362)],
        'Braess_6_4200_10_c1': [16, 0.99, 0.99, (64.81061904761914, 0.012303172983496459, 0.2964392274052476)],
        'Braess_7_4200_10_c1': [16, 0.99, 0.99, (74.91415476190544, 0.010307142857137517, 0.2583309523809412)],
        'BBraess_1_2100_10_c1_2100': [4, 0.98, 0.98, (7.915383219949779, 0.10420607709717421, 0.10539655328765041)],
        'BBraess_3_2100_10_c1_900': [8, 0.99, 0.99, (31.512402380952707, 0.2992891156464251, 0.5405180452975712)],
        'BBraess_5_2100_10_c1_900': [4, 0.99, 0.99, (56.28252698412511, 0.08867499118164085, 0.29346792930926613)],
        'BBraess_7_2100_10_c1_900': [4, 0.99, 0.99, (129.69407936507963, 0.04473189018464778, 0.2595661529789578)],
        'OW': [8, 0.99, 0.99, (80.28925882352935, 0.10513853575962762, 0.4673880208774156)],
        'SF': [10, 0.9997, 0.999, (2168.986889462446, 0.0006119932635893602, 0.0006918085995888627)]
    }

    logstream = ''

    trials = 0
    fails = 0

    for _, net in enumerate(expected_results.keys()):
        trials += 1

        np.random.seed(123456789)

        K = expected_results[net][0]
        alpha_decay = expected_results[net][1]
        epsilon_decay = expected_results[net][2]
        expected_values = expected_results[net][3]

        logstream += f'\n\n\tTesting algorithm TQLearning on network {net}...\t{expected_values}'

        exp = TQLearningExperiment(
            _id=_,
            episodes=10,
            net=net,
            k=K,
            alpha_decay=alpha_decay,
            epsilon_decay=epsilon_decay,
            rep=1
        )

        start_t = time.time()
        res = exp.run_experiment(_)
        end_t = time.time()

        if tuple(res[:3]) != expected_values:
            fails += 1
            logstream += f'\n\tError while validating algorithm TQLearning on network {net}! {res[:3]}'

        logstream += f'\nElapsed time: {end_t - start_t} seconds'

    logstream += '\n\n\tTest completed! Failed trials: %d out of %d (%.1f%%)\n\n' % (fails, trials, (fails/float(trials))*100)

    print(logstream)

    assert fails == 0
