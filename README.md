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
| `routes_costs_sum`               | it stores the sum of costs from each route through an entire experiment                                   |
| `routes_costs_min`               | it stores the min of cost between routes for each OD pair                                                 |

We also implemented custom functions to retrieve information about the routes and drivers of the environment.

| Function                         | Description                                                |
|----------------------------------|------------------------------------------------------------|
| `get_free_flow_travel_times(od)` | get the free flow travel time for every route in a OD pair |
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

For now, we only implemented learning agents for drivers of our environment.

### Networks

Network definitions are at the `/route_choice_env/networks` directory.

| Network                   | OD pairs | Flow (total) | Routes (total) |
|---------------------------|----------|--------------|----------------|
| Braess_1_4200_10_c1       | 1        | 4200         | 3              | 
| Braess_2_4200_10_c1       | 1        | 4200         | 5              |
| Braess_3_4200_10_c1       | 1        | 4200         | 7              |
| Braess_4_4200_10_c1       | 1        | 4200         | 9              |
| Braess_5_4200_10_c1       | 1        | 4200         | 11             |
| Braess_6_4200_10_c1       | 1        | 4200         | 13             |
| Braess_7_4200_10_c1       | 1        | 4200         | 15             |
| BBraess_1_2100_10_c1_2100 | 2        | 4200         | 3              |
| BBraess_3_2100_10_c1_900  | 2        | 4200         | 10             |
| BBraess_5_2100_10_c1_900  | a        | 4200         | 29             |
| BBraess_7_2100_10_c1_900  | 2        | 4200         | 79             |
| OW                        | 4        | 1700         | 64             |

## Running experiments

Single thread:

`$ python3 experiments/experiment_executor.py --alg RMQLearning`

Multiprocess (faster but more expensive):

`$ python3 experiments/experiment_executor.py --alg RMQLearning --workers 6`

## Citing

Citation to appear
