# Route Choice Env: Route Choice Environments for Multiagent Reinforcement Learning.

This repository contains multiagent route choice environments for Multiagent Reinforcement Learning (MARL). The environments follow the [PettingZoo API](https://pettingzoo.farama.org/api/parallel/#).

## Install

To install route-choice-env, follow these steps:

`git clone https://github.com/ramos-ai/route-choice-gym`
`cd route-choice-gym`
`pip install -e .`

## Implemented algorithms

- RMQLearning
- TQLearning

## Examples

Single thread:

`$ python3 experiments/experiment_executor.py --alg RMQLearning`

Multiprocess (faster but more expensive):

`$ python3 experiments/experiment_executor.py --alg RMQLearning --workers 6`

## Citing

Citation to appear
