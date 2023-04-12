# Route Choice Env: Route Choice Environments for Multiagent Reinforcement Learning.

This repository contains multiagent route choice environments for Multiagent Reinforcement Learning (MARL). The environments follow the [PettingZoo API](https://pettingzoo.farama.org/api/parallel/#).

---
## Installing

To install route-choice-env, follow these steps:

`git clone https://github.com/ramos-ai/route-choice-gym`

`cd route-choice-gym`

`pip install -e .`


## Environment

### Usage

Usage examples are available in the `/examples` directory.
There you can find a basic usage of the library to train agents fo the environment.

We encourage you to look for the `rmqlearning_ow_petttingzoo.py` example.
The code presented there is runnable and should give you a overview on how the library works.

### Properties

Properties from the PettingZoo API are:

| Property            | Description                                          |
|---------------------|------------------------------------------------------|
| `observation_space` | `None`                                               |
| `action_space`      | a list of possible routes to take                   |
| `reward`            | a scalar being the travel time of taking a route     |
| `info`              | a dictionary with the keys: [free_flow_travel_times] |

We also provide a set of macroscopic properties from the road network of our environment.
All of these can be accessed directly from our main env class.

| Property                         | Description                                                                                               |
|----------------------------------|-----------------------------------------------------------------------------------------------------------|
| `avg_travel_time`                | the average travel time over the network                                                                  |
| `od_pairs`                       | the network's OD pairs                                                                                    |
| `road_network_flow_distribution` | the flow distribution of drivers over the network. essentially it shows how many agents choose each route |
| `routes_costs_sum`               | the sum of costs in each route for N episodes                                   |
| `routes_costs_min`               | the min cost between routes for each OD pair                                                 |

We also implemented custom functions to retrieve information about the routes and drivers of the environment.

| Function                         | Description                                                |
|----------------------------------|------------------------------------------------------------|
| `get_free_flow_travel_times(od)` | get the free flow travel time for every route of a OD pair |
| `get_driver_flow(d_id)`          | get a driver's flow                                        |
| `get_driver_od_pair(d_id)`       | get a driver's OD pair                                     |


#### Flow distribution

One important property of our environment is the flow distribution.
This property displays the flow distribution of drivers over the network.

It is stored in a special data structure representing OD pairs and routes from the network.

```python
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
```

Consider the list above from a network with 4 OD pairs with 3 routes each and a total flow of 0 distributed across routes.

### Driver agents

Learning algorithms are at the `/route_choice_env/agents` directory.

For now, we only implemented agents for drivers of our environment.

### Networks

Available networks specification can be found at [MASLAB's transportation network repository](https://github.com/maslab-ufrgs/transportation_networks)

## Running experiments

Single thread:

`$ python3 experiments/experiment_executor.py --alg RMQLearning`

Multiprocess (faster but more expensive):

`$ python3 experiments/experiment_executor.py --alg RMQLearning --workers 6`

## Citing

Citation to appear
