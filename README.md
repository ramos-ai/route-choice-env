# Route-Choice-Gym: Multi-Agent RL Route Choice Environment.

## Setup
1. Clone the repository and `cd` into it
2. From the root directory, type `pip install -e .`


## Implemented algorithms
- RMQLearning
- TQLearning


## Running experiments

Single thread:

`$ python3 experiments/experiment_executor.py --alg RMQLearning`


Multiprocess (faster but more expensive):

`$ python3 experiments/experiment_executor.py --alg RMQLearning --workers 6`
