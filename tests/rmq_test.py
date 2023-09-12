import time
import numpy as np

from experiments.rmq_learning_exp import RMQLearningExperiment


def test_rmqlearning_is_consistent():
    expected_results = {
        'Braess_1_4200_10_c1': [4, 0.99, 0.99, (15.658891723347011, 0.10623382653088378, 0.25092952097516913)],
        'Braess_2_4200_10_c1': [4, 0.995, 0.995, (27.48403514738267, 0.12423707482997126, 0.2762391534391447)],
        'Braess_3_4200_10_c1': [4, 0.99, 0.99, (41.36176020407948, 0.09617186791379342, 0.23671728741493236)],
        'Braess_4_4200_10_c1': [4, 0.99, 0.99, (50.020521541952675, 0.004024365079365229, 0.17182945578231312)],
        'Braess_5_4200_10_c1': [8, 0.995, 0.995, (61.610246598639975, 0.0784532738095006, 0.20205906557064923)],
        'Braess_6_4200_10_c1': [8, 0.995, 0.995, (74.66845578231077, 0.053910981535466106, 0.1769212925170026)],
        'Braess_7_4200_10_c1': [8, 0.995, 0.995, (84.52598809523748, 0.028728801020408354, 0.15144270408162336)],
        'BBraess_1_2100_10_c1_2100': [4, 0.98, 0.98, (8.313607709744398, 0.05435776077115378, 0.09211080498880325)],
        'BBraess_3_2100_10_c1_900': [4, 0.995, 0.995, (31.10083412697789, 0.10020349206357661, 0.30219198696146543)],
        'BBraess_5_2100_10_c1_900': [4, 0.995, 0.995, (57.15271190475977, 0.03583497039559119, 0.3000387112622725)],
        'BBraess_7_2100_10_c1_900': [4, 0.995, 0.995, (130.7899936507944, 0.02208753617321256, 0.29788835168988403)],
        'OW': [8, 0.995, 0.995, (81.8699176470589, 0.06055426153195076, 0.18763121878967437)],
        'SF': [4, 0.9999, 0.998, (606.7783587557409, 9.181356693965424e-05, 0.00010139659336468748)]
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

        logstream += f'\n\n\tTesting algorithm RMQLearning on network {net}...\t{expected_values}'

        exp = RMQLearningExperiment(
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
            logstream += f'\n\tError while validating algorithm RMQLearning on network {net}! {res[:3]}'

        logstream += f'\n\tElapsed time: {end_t - start_t} seconds'

    logstream += '\n\n\tTest completed! Failed trials: %d out of %d (%.1f%%)\n\n' % (fails, trials, (fails/float(trials))*100)

    print(logstream)

    assert fails == 0
