from time import time
import numpy as np
from grid_world import GridWorldMultiEnv
from heuristics import HeuristicAgent

env_args = {"size_x": 6,
            "size_y": 6,
            "nr_pickers": 2,
            "n_orders_per_step": 1,
            "action_mode": 'na',    # Changing the action mode does not impact performance of heuristics
            "holding_cost": 1,
            "tardiness_cost": 10,
            "order_probability": 0.8,
            "min_tw": 5,
            "max_tw": 5,
            "max_events": 100,
            }

if __name__ == '__main__':
    env = GridWorldMultiEnv(size_x=env_args['size_x'], size_y=env_args['size_y'], n_agents=env_args['nr_pickers'],
                            n_orders_per_step=env_args['n_orders_per_step'], action_mode=env_args['action_mode'],
                            order_probability=env_args["order_probability"],
                            min_tw=env_args["min_tw"], max_tw=env_args["max_tw"],
                            render_mode="human", max_events=env_args['max_events'])

    print("Environment created")

    env.reset()

    # method = 'random'

    method = 'heur'
    coord = True
    cost_based = True
    heur_agent = HeuristicAgent(env, coord, cost_based)

    rewards = []

    start_time = time()

    n_episodes = 100
    for ep in range(n_episodes):

        start_time_ep = time()

        tot_rewards = {}
        for idx in range(env_args['nr_pickers']):
            tot_rewards[f'Picker_{idx}'] = 0

        n_it = 0
        for agent in env.agent_iter():

            n_it += 1
            observation, reward, termination, truncation, info = env.last()

            order_free_locs = [(o.location, o.state, o.time_windows[o.state]) for o in env.orders if o.assigned == -1]
            order_ass_locs = [(o.location, o.state, o.time_windows[o.state]) for o in env.orders if o.assigned != -1]

            tot_rewards[agent] += reward

            action = None
            if not (termination or termination):
                # RANDOM
                mask = env.get_mask()

                if method == 'random':
                    # NA, EA
                    probs = mask / mask.sum()
                    action = env.np_random.choice(env.size_x * env.size_y, 1, p=probs)                    # NA
                    # action = env.np_random.choice(env.size_x * env.size_y * env.n_agents, 1, p=probs)       # EA

                    # MH
                    # probs_origin = mask[0] / mask[0].sum()
                    # action_origin = env.np_random.choice(env.size_x * env.size_y, 1, p=probs_origin)
                    # mask[1, action_origin] = 1
                    # probs_dest = mask[1] / mask[1].sum()
                    # action_dest = env.np_random.choice(env.size_x * env.size_y, 1, p=probs_dest)
                    # action = (action_origin, action_dest)

                # HEURISTIC
                elif method == 'heur':
                    action = heur_agent.get_action()

            env.step(action)

        rewards.append(np.sum([rew for rew in tot_rewards.values()]) / env_args['nr_pickers'])

        eval_time_time_ep = time() - start_time_ep

        env.reset()

    eval_time = time() - start_time

    print(f"Total time: {eval_time}")
    print(f"Average reward: {np.mean(rewards)} +- {np.std(rewards)}")

    env.close()
