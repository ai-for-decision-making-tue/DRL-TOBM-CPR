import numpy as np
import torch


def preprocess(obs, action_mode):
    if action_mode == 'ea':
        return [{
            "graph_nodes": torch.from_numpy(obs["graph"].nodes).float(),
            "graph_edge_links": torch.from_numpy(obs["graph"].edge_links).int(),
            "agent_locations": torch.from_numpy(obs["locs"]).long(),
            "mask": torch.from_numpy(obs["mask"]).to(torch.int8)
        }]
    else:
        return [{
            "graph_nodes": torch.from_numpy(obs["graph"].nodes).float(),
            "graph_edge_links": torch.from_numpy(obs["graph"].edge_links).int(),
            "mask": torch.from_numpy(obs["mask"]).to(torch.int8)
        }]


def graph_element_to_coordinate(location: int, size_x: int) -> dict:
    row = location // size_x
    return {'row': row,
            'column': location - (row * size_x)}


def coordinate_to_graph_element(row: int, column: int, size_x: int) -> int:
    return column + row * size_x


def distance(origin: int, destination: int, size_x: int) -> int:
    origin_coord = graph_element_to_coordinate(origin, size_x)
    destination_coord = graph_element_to_coordinate(destination, size_x)

    horizontal_dist = abs(origin_coord['row'] - destination_coord['row'])
    vertical_dist = abs(origin_coord['column'] - destination_coord['column'])

    return horizontal_dist + vertical_dist


def get_distance_matrix(size_x, size_y):
    dist_mat = []
    for i in range(size_x * size_y):
        dist_mat.append([])
        for j in range(size_x * size_y):
            if i == j:
                dist_mat[i].append(1)
            else:
                dist_mat[i].append(distance(i, j, size_x))

    return np.array(dist_mat)


def get_edge_list_grid(dist_matrix):
    adj_matrix = (dist_matrix == 1)

    index = adj_matrix.nonzero().transpose(0, 1)
    if len(index) == 3:
        batch = index[0] * adj_matrix.size(-1)
        index_tuple = (batch + index[1], batch + index[2])
    else:
        index_tuple = (index[0], index[1])

    edge_list = np.stack(index_tuple, dim=0)

    return np.array(edge_list)


# Returns true if j is adjacent to i and j is a valid position on the map (e.g., it does not exceed the map's dimensions) and j is not an obstacle
def is_allowed_neighbor(warehouse, i, j):

    # not out of borders
    if j < 0 or j >= warehouse.size_x * warehouse.size_y:
        return False

    # Check maximum manhattan distance
    if distance(i, j, warehouse.size_x) > 1:
        return False

    return True
