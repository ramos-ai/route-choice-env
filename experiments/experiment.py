import os
import contextlib
from concurrent.futures import ProcessPoolExecutor


class Experiment(object):
    """
        Interface for implementing experiments.

        Every method is already defined in this class. It is only needed to implement the run_replication function on
        its subclasses.
    """
    _ID: int
    ALG: str
    ITERATIONS: int
    NET: str
    K: int
    REP: int
    DECAY: float
    ALPHA: float
    ALPHA_DECAY: float
    MIN_ALPHA: float
    EPSILON: float
    EPSILON_DECAY: float
    MIN_EPSILON: float

    def __init__(self, _id: int, algorithm: str, episodes: int, net: str, k: int, decay: float, rep: int):
        self._ID = _id
        self.ALG = algorithm
        self.ITERATIONS = episodes
        self.NET = net
        self.K = k
        self.DECAY = decay
        self.REP = rep
        self.ALPHA = 1.0
        self.ALPHA_DECAY = decay
        self.MIN_ALPHA = 0.0
        self.EPSILON = 1.0
        self.EPSILON_DECAY = decay
        self.MIN_EPSILON = 0.0

        # LOG
        self.LOG_V = '00'
        self.LOGPATH = self.__create_log_path()

    @property
    def results_summary_filename(self):
        return f'{self.LOGPATH}/results_v{self.LOG_V}_summary.txt'

    def run(self):
        results = {}
        for r in range(1, self.REP + 1):
            results[r] = self.run_experiment(r)

        with contextlib.suppress(Exception):
            self.__log_summary(results)

    def run_multiprocess(self, workers):
        futures = {}
        results = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rep in range(1, self.REP + 1):
                futures[rep] = ex.submit(self.run_experiment, r_id=rep)

        for rep, future in futures.items():
            results[rep] = future.result()

        with contextlib.suppress(Exception):
            self.__log_summary(results)

    def run_experiment(self, r_id: int) -> tuple:
        """
        Implement this method returning a tuple of results if you want to log summary of the experiment.
        """
        raise NotImplementedError

    def __log_summary(self, results: dict[int, tuple]):
        with open(self.results_summary_filename, 'a+') as log:
            log.write(f'Results\t{self.ALG}\t{self.NET}\n')
            log.write('rep\tavg-tt\treal\test\tabsdiff\treldiff\n')  # \tproximityUE\n')
            for rep, result in results.items():
                # result is a vector with 5 indices:
                # 0:    avg-tt
                # 1:    real regret
                # 2:    est regret
                # 3:    absolute diff between the est and real regrets
                # 4:    relative diff between the est and real regrets
                log.write(f'{rep}\t{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\t{result[4]}\n')

    def __create_log_path(self):
        logpath = f'{os.path.dirname(os.path.abspath(__file__))}'
        paths = ['/results', f'/{self.ALG}', f'/{self.NET}', f'/K_{self.K}', f'/DECAY_{self.DECAY}']
        for _path in paths:
            logpath += _path
            if not os.path.exists(logpath):
                os.mkdir(logpath)
        return logpath
