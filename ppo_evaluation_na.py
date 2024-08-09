import torch
import numpy as np

import tianshou as ts
from adapted_tianshou.pettingzoo_env import ModifiedPettingZooEnv
from adapted_tianshou.collector import Collector

from pettingzoo.utils import BaseWrapper

from time import time

from gnn_models import NAGNN
from grid_world import GridWorldMultiEnv

from statsmodels.stats.weightstats import DescrStatsW

from utils import get_distance_matrix

env_args = {"size_x": 6,
            "size_y": 6,
            "nr_pickers": 2,
            "n_orders_per_step": 1,
            "action_mode": 'na',
            "holding_cost": 1,
            "tardiness_cost": 10,
            "order_probability": 0.8,
            "min_tw": 5,
            "max_tw": 5,
            "max_events": 100
            }


net_args = {"hidden_dim": 64,
            "lr": 5e-3,
            "discount_factor": 0.99,
            }

device = "cpu"

if __name__ == "__main__":
    def get_env():
        env = GridWorldMultiEnv(size_x=env_args['size_x'], size_y=env_args['size_y'], n_agents=env_args['nr_pickers'],
                                n_orders_per_step=env_args['n_orders_per_step'], action_mode=env_args['action_mode'],
                                order_probability=env_args["order_probability"],
                                min_tw=env_args["min_tw"], max_tw=env_args["max_tw"],
                                render_mode="human", max_events=env_args['max_events'])

        return ModifiedPettingZooEnv(BaseWrapper(env))


    print("Creating envs")

    test_envs = ts.env.DummyVectorEnv(
        [get_env for _ in range(1)]
    )

    dist_matrix = torch.FloatTensor(get_distance_matrix(env_args["size_x"], env_args["size_x"]))
    actor_net = NAGNN(
        8,
        net_args["hidden_dim"],
        env_args["size_x"],
        n_graph_layers=3,   # Set to 0 for inv-ff models
        mode='actor',
        min_val=torch.finfo(torch.float).min
    ).to(device)

    critic_net = NAGNN(
        8,
        net_args["hidden_dim"],
        env_args["size_x"],
        n_graph_layers=3,   # Set to 0 for inv-ff models
        mode='critic',
        min_val=torch.finfo(torch.float).min
    ).to(device).share_memory()

    optim = torch.optim.Adam(
        params=list(actor_net.parameters()) + list(critic_net.parameters()),
        lr=net_args["lr"]
    )

    policy = ts.policy.PPOPolicy(actor_net, critic_net, optim,
                                 discount_factor=net_args["discount_factor"],
                                 value_clip=True,
                                 dist_fn=torch.distributions.categorical.Categorical,
                                 deterministic_eval=True
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


    policy.load_state_dict(
        torch.load(
            "models/ppo_na_gnn.pt"
            # "models/ppo_na_invff.pt"
        )
    )

    policy.eval()
    collector = Collector(policy, test_envs, exploration_noise=False, preprocess_fn=preprocess_function)

    start_time = time()

    result = collector.collect(n_episode=100)
    result['rews'] = result['rews'] / env_args['nr_pickers']
    result['rew'] = result['rew'] / env_args['nr_pickers']

    eval_time = time() - start_time
    print(f'Total evaluation time: {eval_time}')

    print(
        f"Average reward over {len(result['rews'])} episodes: {result['rew']}, std: {np.std(result['rews'])}, CI: {DescrStatsW(result['rews']).tconfint_mean()}")

