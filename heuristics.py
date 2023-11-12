from grid_world import GridWorldMultiEnv
from utils import distance, coordinate_to_graph_element
import functools


class Comparer:
    def __init__(self, cur_loc, grid_size, cost_based, costs):
        self.cur_loc = cur_loc
        self.grid_size = grid_size
        self.cost_based = cost_based
        self.costs = costs

    def __call__(self, l, r):
        if self.cost_based:
            # Both orders are tardy: no state change can occur anymore
            # only distance matters since they will keep incurring the same cost
            if l.state == 2 and r.state == 2:
                l_cost = distance(self.cur_loc, l.location, self.grid_size)
                r_cost = distance(self.cur_loc, r.location, self.grid_size)

            # One or both orders are ongoing: effective costs must be computed keeping track of state changes
            elif l.state == 2 and r.state == 1:
                dl = distance(self.cur_loc, l.location, self.grid_size)
                dr = distance(self.cur_loc, r.location, self.grid_size)
                dlr = distance(l.location, r.location, self.grid_size)
                l_cost = (self.costs[2] * dl + self.costs[1] * min(dl + dlr, r.time_windows[1]) +
                          self.costs[2] * max(0, (dl + dlr) - r.time_windows[1]))
                r_cost = self.costs[2] * (dr + dlr) + self.costs[1] * min(dr, r.time_windows[1]) + self.costs[2] * max(
                    0, dr - r.time_windows[1])

            elif l.state == 1 and r.state == 2:
                dl = distance(self.cur_loc, l.location, self.grid_size)
                dr = distance(self.cur_loc, r.location, self.grid_size)
                dlr = distance(l.location, r.location, self.grid_size)
                l_cost = self.costs[2] * (dl + dlr) + self.costs[1] * min(dl, l.time_windows[1]) + self.costs[2] * max(
                    0, dl - l.time_windows[1])
                r_cost = (self.costs[2] * dr + self.costs[1] * min(dr + dlr, l.time_windows[1]) +
                          self.costs[2] * max(0, (dr + dlr) - l.time_windows[1]))

            elif l.state == 1 and r.state == 1:
                dl = distance(self.cur_loc, l.location, self.grid_size)
                dr = distance(self.cur_loc, r.location, self.grid_size)
                dlr = distance(l.location, r.location, self.grid_size)
                l_cost = (self.costs[1] * min(dl, l.time_windows[1]) + self.costs[2] * max(0, dl - l.time_windows[1]) +
                          self.costs[1] * min(dl + dlr, r.time_windows[1]) + self.costs[2] * max(0, (dl + dlr) -
                                                                                                 r.time_windows[1]))
                r_cost = (self.costs[1] * min(dr, r.time_windows[1]) + self.costs[2] * max(0, dr - r.time_windows[1]) +
                          self.costs[1] * min(dr + dlr, l.time_windows[1]) + self.costs[2] * max(0, (dr + dlr) -
                                                                                                 l.time_windows[1]))

            else:
                # Both orders are announced: announced orders are only compared between each other
                # we try to minimize a combination of distance and time before state change to ongoing
                dl = distance(self.cur_loc, l.location, self.grid_size)
                dr = distance(self.cur_loc, r.location, self.grid_size)
                l_cost = dl + l.time_windows[0]
                r_cost = dr + r.time_windows[0]

            return l_cost - r_cost

        else:
            return distance(self.cur_loc, l.location, self.grid_size) - distance(self.cur_loc, r.location,
                                                                                 self.grid_size)


class HeuristicAgent():
    def __init__(self, env: GridWorldMultiEnv, coord, cost_based):
        self.env = env
        self.coord = coord
        self.cost_based = cost_based

    def get_action(self):
        if self.env.action_mode == 'na':
            cur_agent = self.env.current_picker
        else:
            cur_agent = self.env.decisions_required[0]

        cur_loc = self.env.pickers[cur_agent].location  # Location of current picker
        gridsize = self.env.size_x                      # Assuming a square grid

        comparer = Comparer(cur_loc, self.env.size_x, self.cost_based, [0, self.env.holding_cost, self.env.tardiness_cost])

        # ONGOING AND TARDY ORDERS
        ranked_orders = sorted([o for o in self.env.orders if (o.state >= 1 and o.assigned == -1)],
                               key=functools.cmp_to_key(comparer), reverse=False)

        if self.coord:
            for order in ranked_orders:
                closest = True
                for picker in self.env.pickers:
                    if picker.idx != cur_agent and not picker.allocated:
                        other_loc = picker.location
                        if distance(other_loc, order.location, gridsize) < distance(cur_loc, order.location, gridsize):
                            closest = False
                            break
                if closest:
                    if self.env.action_mode == 'na':
                        return order.location
                    elif self.env.action_mode == 'ea':
                        return coordinate_to_graph_element(cur_agent, order.location,
                                                                 gridsize * gridsize)
                    else:
                        return self.env.pickers[cur_agent].location, order.location
        else:
            if len(ranked_orders) > 0:
                if self.env.action_mode == 'na':
                    return ranked_orders[0].location
                elif self.env.action_mode == 'ea':
                    return coordinate_to_graph_element(cur_agent, ranked_orders[0].location, gridsize * gridsize)
                else:
                    return self.env.pickers[cur_agent].location, ranked_orders[0].location

        # ANNOUNCED ORDERS
        ranked_orders = sorted([o for o in self.env.orders if (o.state == 0 and o.assigned == -1)],
                               key=functools.cmp_to_key(comparer), reverse=False)

        if self.coord:
            for order in ranked_orders:
                closest = True
                for picker in self.env.pickers:
                    if picker.idx != cur_agent and not picker.allocated:
                        other_loc = picker.location
                        if distance(other_loc, order.location, gridsize) < distance(cur_loc, order.location, gridsize):
                            closest = False
                            break
                if closest:
                    if self.env.action_mode == 'na':
                        return order.location
                    elif self.env.action_mode == 'ea':
                        return coordinate_to_graph_element(cur_agent, order.location,
                                                           gridsize * gridsize)
                    else:
                        return self.env.pickers[cur_agent].location, order.location
        else:
            if len(ranked_orders) > 0:
                if self.env.action_mode == 'na':
                    return ranked_orders[0].location
                elif self.env.action_mode == 'ea':
                    return coordinate_to_graph_element(cur_agent, ranked_orders[0].location, gridsize * gridsize)
                else:
                    return self.env.pickers[cur_agent].location, ranked_orders[0].location

        # NO ORDERS, STAY IN PLACE
        if self.env.action_mode == 'na':
            return cur_loc
        elif self.env.action_mode == 'ea':
            return coordinate_to_graph_element(cur_agent, cur_loc, gridsize * gridsize)
        else:
            return cur_loc, cur_loc
