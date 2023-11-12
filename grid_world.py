import numpy as np

import torch
from gymnasium import spaces
from gymnasium.spaces import GraphInstance
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from torch_geometric.utils import dense_to_sparse

from utils import graph_element_to_coordinate, coordinate_to_graph_element, \
    distance, get_distance_matrix, is_allowed_neighbor


class GridWorldMultiEnv(AECEnv):

    metadata = {"render_modes": ["human"], "name": "gridworld_multi"}

    def __init__(self, size_x, size_y, n_agents, n_orders_per_step, action_mode,
                 holding_cost=1, move_cost=0, tardiness_cost=10, order_probability=0.8,
                 min_tw=5, max_tw=5, max_events=1000, use_dispatch_locations=False,
                 fixed_initial_state=False, manual_initial_state=False, fixed_events=None,
                 clairvoyant_sol_ready=False, render_mode=None):

        self.np_random = None

        self.size_x = size_x
        self.size_y = size_y
        self.n_agents = n_agents
        self.n_orders_per_step = n_orders_per_step

        # na, ea, mh
        self.action_mode = action_mode

        self.holding_cost = holding_cost
        self.move_cost = move_cost
        self.tardiness_cost = tardiness_cost

        self.order_probability = order_probability
        self.min_tw = min_tw
        self.max_tw = max_tw

        self.max_events = max_events
        self.n_events = 0

        self.use_dispatch_locations = use_dispatch_locations

        self.fixed_initial_state = fixed_initial_state
        self.manual_initial_state = manual_initial_state
        self.fixed_events = fixed_events

        self.clairvoyant_sol_ready = clairvoyant_sol_ready

        self.distance_matrix = get_distance_matrix(size_x, size_y)
        # self.edge_list = dense_to_sparse(torch.tensor(self.distance_matrix))[0].detach().numpy()
        self.edge_list_grid = dense_to_sparse(torch.tensor(self.distance_matrix == 1))[0].detach().numpy()

        # MULTI AGENT APPROACH
        self.possible_agents = [f"Picker_{idx}" for idx in range(n_agents)]

        # Agent mapping needed for ea and mh approach, since agents do not correspond to pickers anymore:
        # e.g.: At initialization we have {a0:p0, a1:p1}, both p0 and p1 require decision
        # Agents are looped in order a0, a1. If p1 is selected first and p0 second, then we have {a0: p1, a1:p0}
        # It is important to keep track of this for proper credit assignment
        self.agent_name_mapping = {f"Picker_{idx}": idx for idx in range(n_agents)}

        if self.action_mode == 'na':
            # size_x * size_y = n_nodes actions, corresponding to possible destination nodes
            self.action_spaces = {agent: spaces.Discrete(self.size_x * self.size_y) for agent in self.possible_agents}
            self.n_node_features = 8

            self.observation_spaces = {agent: spaces.Dict({
                "graph": spaces.Graph(
                    node_space=spaces.Box(low=0, high=1000, shape=(self.n_node_features,), dtype=np.float64),
                    edge_space=spaces.Discrete(self.size_x * self.size_y)),
                "mask": spaces.MultiBinary(self.size_x * self.size_y)})
                for agent in self.possible_agents}

        elif self.action_mode == 'ea':
            # n_agents * size_x * size_y = n_agents * n_nodes actions, corresponding to possible agent-destination combinations
            self.action_spaces = {agent: spaces.Discrete(self.n_agents * self.size_x * self.size_y) for agent in self.possible_agents}
            self.n_node_features = 6

            self.observation_spaces = {agent: spaces.Dict({
                "graph": spaces.Graph(
                    node_space=spaces.Box(low=0, high=1000, shape=(self.n_node_features,), dtype=np.float64),
                    edge_space=spaces.Discrete(self.size_x * self.size_y)),
                "locs": spaces.Sequence(spaces.Discrete(self.size_x * self.size_y)),
                "mask": spaces.MultiBinary(self.n_agents * self.size_x * self.size_y)})
                for agent in self.possible_agents}

        elif self.action_mode == 'mh':
            # (size_x * size_y, size_x * size_y) = (n_nodes, n_nodes) tuple actions, corresponding to origin, destination couples
            self.action_spaces = {agent: spaces.MultiDiscrete([self.size_x * self.size_y, self.size_x * self.size_y]) for agent in self.possible_agents}
            self.n_node_features = 6

            self.observation_spaces = {agent: spaces.Dict({
                "graph": spaces.Graph(
                    node_space=spaces.Box(low=0, high=1000, shape=(self.n_node_features,), dtype=np.float64),
                    edge_space=spaces.Discrete(self.size_x * self.size_y)),
                "mask": spaces.MultiBinary(self.size_x * self.size_y)})
                for agent in self.possible_agents}

        elif self.action_mode == 'mh_inv':
            # (size_x * size_y, size_x * size_y) = (n_nodes, n_nodes) tuple actions, corresponding to origin, destination couples
            self.action_spaces = {agent: spaces.MultiDiscrete([self.size_x * self.size_y, self.size_x * self.size_y]) for agent in self.possible_agents}
            self.n_node_features = 6

            self.observation_spaces = {agent: spaces.Dict({
                "graph": spaces.Graph(
                    node_space=spaces.Box(low=0, high=1000, shape=(self.n_node_features,), dtype=np.float64),
                    edge_space=spaces.Discrete(self.size_x * self.size_y)),
                "order_mask": spaces.MultiBinary(self.size_x * self.size_y),
                "mask": spaces.MultiBinary(self.size_x * self.size_y)})
                for agent in self.possible_agents}

    def get_mask(self):
        if self.action_mode == 'na':
            return self._get_mask_na()
        elif self.action_mode == 'ea':
            return self._get_mask_ea()
        elif self.action_mode == 'mh':
            return self._get_mask_mh()
        elif self.action_mode == 'mh_inv':
            return self._get_mask_mh_inv()

    def _get_mask_na(self):

        mask = np.zeros((self.size_x * self.size_y))

        mask[self.pickers[self.current_picker].location] = 1

        # Only location of orders, since we only care about the destination for a pre-selected picker
        order_locs = [order.location for order in self.orders if order.assigned == -1]
        mask[order_locs] = 1

        return mask

    def _get_mask_ea(self):

        # Mask over (picker_loc, destination_loc) tuples (i.e. edges)
        mask = np.zeros((self.n_agents, self.size_x * self.size_y))

        # Unallocated pickers location always allowed, since pickers can always stay in place
        agent_locs = [picker.location for picker in self.pickers if picker.idx in self.decisions_required]
        mask[self.decisions_required, agent_locs] = 1

        # Location of orders allowed for idle agents
        decision_mask = np.array(self.decisions_required)[..., np.newaxis]
        order_locs = [order.location for order in self.orders if order.assigned == -1]
        mask[decision_mask, order_locs] = 1
        mask = mask.flatten()

        return mask

    def _get_mask_mh(self):
        mask = np.zeros((2, self.size_x * self.size_y))

        # Mask for picker/origin head
        agent_locs = [picker.location for picker in self.pickers if picker.idx in self.decisions_required]
        mask[0][agent_locs] = 1

        # Mask for destination head
        # Only location of orders, the location of the selected picker will be added inside the network
        order_locs = [order.location for order in self.orders if order.assigned == -1]
        mask[1][order_locs] = 1

        return mask

    def _get_mask_mh_inv(self):
        mask = np.zeros((3, self.size_x * self.size_y))

        # Allowed destinations: all agents and orders locations
        order_locs = [order.location for order in self.orders if order.assigned == -1]
        agent_locs = [picker.location for picker in self.pickers if picker.idx in self.decisions_required]
        dest_locs = list(set(order_locs + agent_locs))
        mask[0][dest_locs] = 1

        mask[1][agent_locs] = 1
        mask[2][order_locs] = 1

        return mask

    def _get_obs(self) -> dict:

        node_features = np.zeros((self.size_x * self.size_y, self.n_node_features))

        n = 0
        if self.action_mode == 'na':
            # Current agent
            node_features[self.pickers[self.current_picker].location, 0] = 1

            # Distances from location of current agent
            node_features[:, 7] = self.current_distances

            unallocated_agents_locs = [picker.location for picker in self.pickers if (not picker.allocated and picker.idx != self.current_picker)]
            allocated_agents_locs = [picker.location for picker in self.pickers if (picker.allocated and picker.idx != self.current_picker)]
            allocated_agents_dests = np.array([(picker.destination, distance(picker.location, picker.destination, self.size_x))
                                               for picker in self.pickers if (picker.allocated and picker.idx != self.current_picker)])

            n += 1

        else:
            unallocated_agents_locs = [picker.location for picker in self.pickers if not picker.allocated]
            allocated_agents_locs = [picker.location for picker in self.pickers if picker.allocated]
            allocated_agents_dests = np.array([(picker.destination, distance(picker.location, picker.destination, self.size_x))
                                               for picker in self.pickers if picker.allocated])

        # Unallocated agents
        if len(unallocated_agents_locs) > 0:
            node_features[:, n] = np.bincount(unallocated_agents_locs, minlength=np.size(node_features[:, n]))
        n += 1

        # Allocated agents
        if len(allocated_agents_locs) > 0:
            node_features[:, n] = np.bincount(allocated_agents_locs, minlength=np.size(node_features[:, n]))
        n += 1

        # Destinations
        if len(allocated_agents_locs) > 0:
            node_features[allocated_agents_dests[:, 0], n] = allocated_agents_dests[:, 1]
        n += 1

        # Announced orders
        announced_orders = np.array([(order.location, order.time_windows[0]) for order in self.orders if order.state == 0])
        if announced_orders.shape[0] > 0:
            node_features[announced_orders[:, 0], n] = announced_orders[:, 1]
        n += 1

        # Ongoing orders
        ongoing_orders = np.array([(order.location, order.time_windows[1]) for order in self.orders if order.state == 1])
        if ongoing_orders.shape[0] > 0:
            node_features[ongoing_orders[:, 0], n] = ongoing_orders[:, 1]
        n += 1

        # Tardy orders
        tardy_orders = np.array([(order.location, order.time_windows[2] + 1) for order in self.orders if order.state == 2])
        if tardy_orders.shape[0] > 0:
            node_features[tardy_orders[:, 0], n] = tardy_orders[:, 1]
        n += 1

        mask = self.get_mask()

        obs = {"graph": GraphInstance(
                            nodes=node_features,
                            edges=(self.distance_matrix == 1).flatten(),
                            edge_links=self.edge_list_grid),
               "mask": mask}

        if self.action_mode == 'ea':
            agents_locs = np.array([picker.location for picker in self.pickers])
            obs["locs"] = agents_locs

        return obs

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up-to-date possible)
        at any time after reset() is called.
        """

        return self._get_obs()

    def _get_info(self):
        return dict()

    def reset(self, seed=None, return_info=True, options=None):

        # We need the following line to seed self.np_random
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)
        else:
            self.np_random, _ = seeding.np_random()

        self.n_events = 0

        self.agents = self.possible_agents

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.final_cycle = False

        self.orders = []
        initial_order_locs = []

        self.pickers = []
        self.decisions_required = []
        self.decisions_taken = []
        if self.manual_initial_state:
            self.decisions_required = [0, 1]

            self.pickers.append(Picker(self, 0, 26))
            self.pickers.append(Picker(self, 1, 9))

            # Easy case
            self.orders.append(Order(1, 1, [0, 10, 0]))
            self.orders.append(Order(11, 1, [0, 10, 0]))

            # Hard case
            # self.orders.append(Order(1, 1, [0, 5, 0]))
            # self.orders.append(Order(11, 2, [0, 0, 1]))
            # self.orders.append(Order(34, 0, [2, 10, 0]))

        else:
            # Initialize agents
            for idx in range(self.n_agents):
                # Choose the agent's location uniformly at random
                # TODO: handle possible obstacles
                init_loc = self.np_random.integers(0, self.size_x * self.size_y, size=1, dtype=int).item()

                # Create agent and add it to agent list
                self.pickers.append(Picker(self, idx, init_loc))

                # Initially, every agent requires a new action
                self.decisions_required.append(idx)

            # Initialize orders randomly
            n_initial_orders = self.np_random.integers(0, self.size_x, size=1, dtype=int).item()
            for idx in range(n_initial_orders):
                # Generate different locations for each initial order
                loc = self.np_random.integers(0, self.size_x * self.size_y, size=1, dtype=int).item()
                while loc in initial_order_locs:
                    loc = self.np_random.integers(0, self.size_x * self.size_y, size=1, dtype=int).item()
                initial_order_locs.append(loc)

                state = self.np_random.integers(0, 2, size=1, dtype=int).item()

                t0 = 0
                t1 = 0
                t2 = 0
                if state == 0:
                    t0 = self.np_random.integers(1, self.min_tw + self.max_tw + 1, size=1, dtype=int).item()
                    t1 = self.np_random.integers(self.min_tw, self.min_tw + self.max_tw + 1, size=1, dtype=int).item()
                elif state == 1:
                    t1 = self.np_random.integers(1, self.min_tw + self.max_tw + 1, size=1, dtype=int).item()
                self.orders.append(Order(loc, state, [t0, t1, t2]))

        self.assigned_orders_locs = []

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        if self.action_mode == 'na':
            self.current_picker = 0
        else:
            self.current_picker = -1

        if self.action_mode == 'na':
            self.current_distances = self.distance_matrix[self.pickers[self.current_picker].location]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        agent = self.agent_selection

        if (
            self.terminations[self.agent_selection]
        ):
            self.current_picker = -1
            if self.action_mode == 'na':
                if len(self.agents) > 0:
                    # Await decision for next agent in list
                    self.current_picker = self.decisions_required[0]

                    # Set distanced from current agent location
                    self.current_distances = self.distance_matrix[self.pickers[self.current_picker].location]

                else:
                    self.current_distances = None

            if self._agent_selector.is_last():
                self.agents = []
                self._agent_selector = agent_selector(self.agents)
            else:
                self.agent_selection = self._agent_selector.next()
            return

        # The agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # if self.agent_name_mapping[agent] in self.decisions_required:
        if len(self.decisions_required) > 0:

            mask = self.get_mask()

            if self.action_mode in ['na', 'ea']:
                # Get action as int value
                if hasattr(action, "__len__"):
                    action = action.item()

                if not mask[action]:
                    print(f'Action {action} not allowed for picker {self.current_picker}!')

            elif self.action_mode == 'mh':
                if hasattr(action[0], "__len__"):
                    action = (action[0].item(), action[1].item())
                mask = self.get_mask()
                mask_origin = mask[0]
                mask_dest = mask[1]
                mask_dest[action[0]] = 1.0
                if not mask_origin[action[0]] or not mask_dest[action[1]]:
                    print('Action not allowed!')

            else:
                if hasattr(action[0], "__len__"):
                    action = (action[0].item(), action[1].item())
                mask = self.get_mask()
                mask_origin = mask[1]
                mask_dest = mask[0]
                if not mask_origin[action[0]] or not mask_dest[action[1]]:
                    print('Action not allowed!')

            # apply action
            if self.action_mode == 'na':
                destination = action
                self.pickers[self.current_picker].destination = destination

            elif self.action_mode == 'ea':
                # Decompose edge: row = selected agent id, col = destination node
                incident_nodes = graph_element_to_coordinate(action, self.size_x * self.size_y)

                agent_id = incident_nodes['row']
                destination = incident_nodes['column']

                self.current_picker = agent_id
                self.pickers[self.current_picker].destination = destination

                # Change PettingZoo and problem agent's matching
                self.agent_name_mapping[agent] = self.current_picker

            elif self.action_mode in ['mh', 'mh_inv']:
                self.current_picker = [picker.idx for picker in self.pickers if
                                       picker.location == action[0] and picker.idx in self.decisions_required][0]
                destination = action[1]
                self.pickers[self.current_picker].destination = destination

                # Change PettingZoo and internal agent's matching
                self.agent_name_mapping[agent] = self.current_picker

            else:
                raise "Action mode not implemented!"

            for order in self.orders:
                # Assign order to an agent if the agent is moving there
                # Only if it is not assigned yet (to make stay in place action always possible and prevent deadlocks)
                if order.assigned == -1 and order.location == destination and order.state < 3:
                    self.pickers[self.current_picker].allocated = True
                    self.assigned_orders_locs.append(order.location)
                    order.assigned = self.current_picker
                    break

            try:
                self.decisions_required.remove(self.current_picker)
            except Exception as e:
                raise Exception(f'Error: decision taken for agent {self.current_picker}, but not required!')

        # ENV STEP
        if self._agent_selector.is_last():

            reward = 0
            while len(self.decisions_required) == 0 and not self.final_cycle:
                # Create random event to modify the environment
                event = self._get_event()
                self.n_events += 1

                # Negative reward to minimize costs
                reward -= self._apply_event(event)

                if self.manual_initial_state and self.n_events == self.max_events:
                    break

            # rewards for all agents are placed in the .rewards dictionary
            for agent in self.possible_agents:
                self.rewards[agent] = reward

            # Adds .rewards to ._cumulative_rewards
            self._accumulate_rewards()

            # No rewards are allocated until both players give an action
            self._clear_rewards()

            # Even if it is the last cycle, take "fake" step only for agents that should actually see observation and reward
            self.agents = [agent for agent in self.possible_agents if self.agent_name_mapping[agent] in self.decisions_required]

            if self.n_events >= self.max_events:

                if len(self.agents) == 0 and not self.final_cycle:
                    self.final_cycle = True

                if self.final_cycle:
                    self.terminations = {
                        agent: True for agent in self.possible_agents
                    }

                    # Take last step for all agents, so that rewards are considered (only during test!)
                    self.agents = self.possible_agents
                    self.decisions_required = list(self.agent_name_mapping.values())

                else:
                    self.final_cycle = True

            self._agent_selector = agent_selector(self.agents)

        self.current_picker = -1
        if self.action_mode == 'na':
            if len(self.decisions_required) > 0:
                # Await decision for next agent in list
                self.current_picker = self.decisions_required[0]

                # Set distanced from current agent location
                self.current_distances = self.distance_matrix[self.pickers[self.current_picker].location]

            else:
                self.current_distances = None

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()

    def get_potential_moves(self, location):
        potential_moves = []

        coord = graph_element_to_coordinate(location, self.size_x)

        potential_moves.append(coordinate_to_graph_element(coord['row'], coord['column'], self.size_x))  # stay
        potential_moves.append(coordinate_to_graph_element(coord['row'] - 1, coord['column'], self.size_x))  # up
        potential_moves.append(coordinate_to_graph_element(coord['row'] + 1, coord['column'], self.size_x))  # down
        potential_moves.append(coordinate_to_graph_element(coord['row'], coord['column'] - 1, self.size_x))  # left
        potential_moves.append(coordinate_to_graph_element(coord['row'], coord['column'] + 1, self.size_x))  # right

        return potential_moves

    def _get_holding_costs(self):

        ongoing_orders = 0
        tardy_orders = 0
        for order in self.orders:
            if order.state == 1:
                ongoing_orders += 1
            elif order.state == 2:
                tardy_orders += 1

        return ongoing_orders * self.holding_cost + tardy_orders * self.tardiness_cost

    def _manage_orders(self):

        to_delete = []
        for idx, o in enumerate(self.orders):

            old_state = o.state

            o.evolve()

            if o.canceled:  # Can only happen in state announced
                if o.assigned >= 0:
                    self.assigned_orders_locs.remove(o.location)
                    self.pickers[o.assigned].allocated = False
                to_delete.append(idx)
            elif old_state == 0 and o.state == 1:
                # Agent picks order if it evolves while it is on it, then delete order and deallocate related agent
                if o.assigned >= 0:
                    picker = self.pickers[o.assigned]
                    if picker.allocated and picker.location == picker.destination == o.location:
                        self.assigned_orders_locs.remove(o.location)
                        picker.allocated = False

                        # Orders picked before due date will stay on board until due date to handle clairvoyant solution event preprocessing
                        if self.clairvoyant_sol_ready:
                            # Change order state
                            o.state = 3
                            o.assigned = -1
                        else:
                            to_delete.append(idx)

        for idx in sorted(to_delete, reverse=True):
            del self.orders[idx]

        return

    def _get_event(self) -> dict:

        locations = []
        time_windows = []

        if self.fixed_events is not None:
            if len(self.fixed_events) > 0:
                event = self.fixed_events.pop(0)
            else:
                raise RuntimeError("No more events to process.")

            if self.order_probability > 0.0:
                locations.append(event[0])
            else:
                locations.append(self.size_x * self.size_y)
            time_windows.append((event[1], event[2]))

        else:
            for n in range(self.n_orders_per_step):
                order_prob = self.np_random.integers(0, 100, size=1, dtype=int).item()
                if order_prob < self.order_probability * 100:
                    locations.append(self.np_random.integers(0, self.size_x * self.size_y, size=1, dtype=int).item())
                else:
                    locations.append(self.size_x * self.size_y)
                time_windows.append((self.np_random.integers(self.min_tw, self.min_tw + self.max_tw + 1, size=1, dtype=int).item(),
                                     self.np_random.integers(self.min_tw, self.min_tw + self.max_tw + 1, size=1, dtype=int).item()))

        event = {
            'locations': locations,
            'time_windows': time_windows
        }
        return event

    def _apply_event(self, event):

        moving_cost = 0

        # Before applying the event, move the agents
        for picker in self.pickers:
            picker.apply_move(self, picker.get_best_move(self))

        # Holding costs are computed before evolution
        holding_costs = self._get_holding_costs()

        # Remove canceled orders and pick potential orders becoming ongoing
        self._manage_orders()

        # New orders are announced on the board
        for i in range(self.n_orders_per_step):
            if event['locations'][i] != self.size_x * self.size_y:
                tws = [event['time_windows'][i][0], event['time_windows'][i][1], 0]
                new_order = Order(event['locations'][i], 0, tws)    # State = 0: orders are always announced before becoming pickable

                # Need to check if no other order is at the location
                if len(self.orders) == 0:
                    self.orders.append(new_order)
                else:
                    flag_already_there = False
                    for order in self.orders:
                        if order.location == new_order.location:
                            flag_already_there = True
                            break
                    if not flag_already_there:
                        self.orders.append(new_order)

        # If all the agents are allocated to an order, no need to take any action
        if len(self.assigned_orders_locs) != self.n_agents:
            for picker in self.pickers:
                if not picker.allocated:
                    self.decisions_required.append(picker.idx)

        return moving_cost + holding_costs

    def close(self):
        pass


class Order():

    def __init__(self, location, state, time_windows):
        self.location = location
        self.state = state
        self.time_windows = time_windows

        self.assigned = -1
        self.tardy = False if self.state < 2 else True
        self.canceled = False

    def evolve(self):
        if self.state == 0:

            # decrease time_window
            self.time_windows[0] -= 1

            # the order is becoming ongoing
            if self.time_windows[0] == 0:
                self.state = 1

        elif self.state == 1:
            # decrease time_window
            self.time_windows[1] -= 1

            # the order is becoming tardy
            if self.time_windows[1] == 0:
                self.tardy = True
                self.state = 2

        elif self.state == 2:
            # increase time_window
            self.time_windows[2] += 1

        elif self.state == 3:
            self.time_windows[1] -= 1
            if self.time_windows[1] == 0:
                self.canceled = True


class Picker():

    def __init__(self, warehouse, idx, location):
        self.idx = idx
        self.warehouse = warehouse
        self.location = location
        self.destination = location
        self.allocated = False

    def get_best_move(self, warehouse):
        potential_moves = warehouse.get_potential_moves(self.location)
        current_best_dist = warehouse.size_x * warehouse.size_y

        current_best_move = self.location

        for move in potential_moves:
            if is_allowed_neighbor(warehouse, self.location, move):
                dist = distance(move, self.destination, warehouse.size_x)
                if dist < current_best_dist:
                    current_best_dist = dist
                    current_best_move = move

        return current_best_move

    def apply_move(self, warehouse, move):
        # Move to new location
        self.location = move

        # Pick potential order
        to_delete = []
        for idx, o in enumerate(warehouse.orders):
            if o.location == move and (o.state == 1 or o.state == 2) and o.assigned == self.idx:

                # Delete order (location) from assigned list

                warehouse.assigned_orders_locs.remove(move)

                # Orders picked before due date will stay on board until due date to handle clairvoyant solver event preprocessing
                if warehouse.clairvoyant_sol_ready and o.state == 1:
                    # Change order state
                    o.state = 3
                    o.assigned = -1
                else:
                    # Delete order from the board
                    to_delete.append(idx)

                # Free agent
                self.allocated = False

                # Return no moving costs because picking happened

        for idx in sorted(to_delete, reverse=True):
            del warehouse.orders[idx]

        return warehouse.move_cost
