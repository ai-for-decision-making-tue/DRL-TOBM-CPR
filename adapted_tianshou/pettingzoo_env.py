from abc import ABC

from pettingzoo import AECEnv
from tianshou.env import PettingZooEnv

from typing import Any, Dict, Tuple

from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseWrapper


class ModifiedPettingZooEnv(PettingZooEnv):
    def __init__(self, env: BaseWrapper):
        super().__init__(env)

    def step(self, action: Any) -> Tuple[Dict, float, bool, bool, Dict]:

        self.env.step(action)

        observation, rew, term, trunc, info = self.env.last()

        if isinstance(observation, dict) and 'action_mask' in observation:
            obs = {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask':
                    [True if obm == 1 else False for obm in observation['action_mask']]
            }
        else:
            if isinstance(self.action_space, spaces.Discrete):
                obs = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                    'mask': [True] * self.env.action_space(self.env.agent_selection).n
                }
            else:
                obs = {'agent_id': self.env.agent_selection, 'obs': observation}

        return obs, rew, term, trunc, info
