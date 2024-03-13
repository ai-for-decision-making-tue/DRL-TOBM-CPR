# Deep Reinforcement Learning for Two-sided Online Bipartite Matching in Collaborative Order Picking
This repository contains the code for the paper "Deep Reinforcement Learning for Two-sided Online Bipartite Matching in Collaborative Order Picking", accepted in the conference track of ACML2023.
https://proceedings.mlr.press/v222/begnardi24a/begnardi24a.pdf

## Requirements
The following libraries are necessary to run the code:
- torch==1.12.1
- gymnasium==0.28.1
- pettingzoo==1.22.3
- tianshou==0.5.0
- pytorch-geometric==2.1.0

## Directory adapted_tianshou
The adapted_tianshou directory contains files adapted from the original library.
In particular:

- `pettingzoo_env`:
The setting is not actually multi-agent, since decisions are taken sequentially from a centralized controller.
We use PettingZoo only to keep track of rewards since we deal with actions with different (multistep) durations.
To do this, the environment needs to return a single reward instead of a vector of rewards.

- `collector_multiagent`:
For the same reason as before, the training collector needs to be modified to store samples collected by multiple (virtual) agents with different durations.

- `pg_multihead`, `a2c_multihead`, `ppo_multihead`:
Necessary to handle action distribution sampling in the multi-head action pipeline during training.

## Usage
- The file grid_world.py contains the simulation environment used to train and test the models;
- The files ppo_training_xx.py, where xx in ['na', 'ea', 'mh'] represents the action space, can be used to train a new model;
- The files ppo_evaluation_xx.py, where xx is the same as above, can be used to test the trained models;
- The file heuristics.py contains the implementation of the heuristics used to benchmark our methods.
- The directory models contains the trained models evaluated in the paper.
