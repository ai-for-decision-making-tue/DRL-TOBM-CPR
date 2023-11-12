from pettingzoo.utils import BaseWrapper

from adapted_tianshou.collector_multiagent import MultiAgentCollector
from gnn_models import MHGNN

from adapted_tianshou.ppo_multihead import PPOPolicy
from adapted_tianshou.pettingzoo_env import ModifiedPettingZooEnv

import os
import tianshou as ts
from grid_world import GridWorldMultiEnv
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import torch

from torch.optim.lr_scheduler import ExponentialLR

from utils import get_distance_matrix


env_args = {"size_x": 6,
            "size_y": 6,
            "nr_pickers": 2,
            "n_orders_per_step": 1,
            "action_mode": 'mh',
            "holding_cost": 0.1,
            "tardiness_cost": 1,
            "order_probability": 0.8,
            "min_tw": 5,
            "max_tw": 5,
            "max_events": 25
            }

train_args = {"hidden_dim": 64,
              "lr": 5e-3,
              "discount_factor": 0.99,
              "batch_size": 64,
              "max_batch_size_ppo": 0,
              "nr_envs": 4,
              "max_epoch": 30,
              "step_per_collect": 50000,
              "step_per_epoch": 100000,
              "repeat_per_collect": 2
              }


def save_best_fn(policy):
    print("Saving improved policy")
    torch.save(policy.state_dict(), f"models/ppo_mh_gnn.pt")
    # torch.save(policy.state_dict(), f"models/ppo_mh_invff.pt")


warehouse = GridWorldMultiEnv(env_args["size_x"], env_args["size_y"], env_args["nr_pickers"],
                              env_args["n_orders_per_step"], env_args["action_mode"],
                              holding_cost=env_args['holding_cost'], tardiness_cost=env_args['tardiness_cost'],
                              max_events=env_args['max_events'])

device = "cpu"

if __name__ == "__main__":
    def get_env():
        env = GridWorldMultiEnv(env_args["size_x"], env_args["size_y"], env_args["nr_pickers"],
                                env_args["n_orders_per_step"], env_args["action_mode"],
                                holding_cost=env_args['holding_cost'], tardiness_cost=env_args['tardiness_cost'],
                                max_events=env_args['max_events'])

        return ModifiedPettingZooEnv(BaseWrapper(env))


    print("Creating envs")

    envs = ts.env.SubprocVectorEnv(
        [get_env for _ in range(train_args["nr_envs"])]
    )
    test_envs = ts.env.DummyVectorEnv(
        [get_env for _ in range(train_args["nr_envs"])]
    )

    dist_matrix = torch.FloatTensor(get_distance_matrix(env_args["size_x"], env_args["size_x"]))
    actor_net = MHGNN(
        6,
        train_args["hidden_dim"],
        env_args["size_x"],
        dist_matrix=dist_matrix,
        n_graph_layers=3,
        mode='actor',
        min_val=torch.finfo(torch.float).min
    ).to(device)

    critic_net = MHGNN(
        6,
        train_args["hidden_dim"],
        env_args["size_x"],
        dist_matrix=dist_matrix,
        n_graph_layers=3,
        mode='critic',
        min_val=torch.finfo(torch.float).min
    ).to(device).share_memory()

    optim = torch.optim.Adam(
        params=list(actor_net.parameters()) + list(critic_net.parameters()),
        lr=train_args["lr"]
    )

    scheduler = ExponentialLR(optim, 0.90)

    policy = PPOPolicy(actor_net, critic_net, optim,
                       discount_factor=train_args["discount_factor"],
                       value_clip=True,
                       dist_fn=torch.distributions.categorical.Categorical,
                       deterministic_eval=True,
                       lr_scheduler=scheduler
                       )
    policy.action_type = "discrete"

    def preprocess_function(**kwargs):
        if "obs" in kwargs:
            obs_with_tensors = [
                {"graph_nodes": torch.from_numpy(obs['obs']["graph"].nodes).float(),
                 "graph_edge_links": torch.from_numpy(obs['obs']["graph"].edge_links).int(),
                 "mask": torch.from_numpy(obs['obs']["mask"]).to(torch.int8),
                 "agent_id": obs['agent_id']}
                for obs in kwargs["obs"]]
            kwargs["obs"] = obs_with_tensors
        if "obs_next" in kwargs:
            obs_with_tensors = [
                {"graph_nodes": torch.from_numpy(obs['obs']["graph"][0]).float(),
                 "graph_edge_links": torch.from_numpy(obs['obs']["graph"][2]).int(),
                 "mask": torch.from_numpy(obs['obs']["mask"]).to(torch.int8),
                 "agent_id": obs['agent_id']}
                for obs in kwargs["obs_next"]]
            kwargs["obs_next"] = obs_with_tensors
        return kwargs


    collector = MultiAgentCollector(policy, envs,
                                    ts.data.VectorReplayBuffer(20000, train_args["nr_envs"]),
                                    exploration_noise=True, preprocess_fn=preprocess_function)
    collector.reset()

    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False, preprocess_fn=preprocess_function)
    test_collector.reset()

    log_path = os.path.join("logs/ppo_mh")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    print("Starting training")
    policy.train()
    result = ts.trainer.onpolicy_trainer(
        policy, collector, test_collector=test_collector,
        max_epoch=train_args["max_epoch"], step_per_epoch=train_args["step_per_epoch"],
        step_per_collect=train_args["step_per_collect"],
        episode_per_test=100, batch_size=train_args["batch_size"],
        repeat_per_collect=train_args["repeat_per_collect"],
        train_fn=lambda epoch, env_step: None,
        save_best_fn=save_best_fn,
        stop_fn=None, show_progress=True, verbose=True,
        test_in_train=True, logger=logger)
    print(f'Finished training! Use {result["duration"]}')
