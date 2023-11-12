import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, LayerNorm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool


class NAGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, gridsize: int, n_graph_layers: int, mode: str, min_val: float):

        super().__init__()

        self.gridsize = gridsize
        self.num_features = input_dim

        # Actor or critic net
        self.mode = mode

        self.n_layers = n_graph_layers

        self.min_val = min_val

        self.convs = torch.nn.ModuleList()

        for _ in range(n_graph_layers):
            mlp = Sequential(
                Linear(input_dim, hidden_dim),
                LayerNorm(hidden_dim),
                ReLU(),
                # Linear(hidden_channels, hidden_channels),
            )
            # fully connected graph contains self-loops:
            # eps=-1 avoids considering nodes' features twice as they are already in the neighborhood
            self.convs.append(GINConv(mlp, eps=-1).jittable())
            input_dim = hidden_dim

        self.mid_dim = self.num_features + n_graph_layers * hidden_dim

        self.lin1 = Linear(self.mid_dim, 2 * hidden_dim)
        self.batch_norm = BatchNorm1d(2 * hidden_dim)
        self.relu = ReLU()
        self.lin2 = Linear(2 * hidden_dim, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, observations: torch.Tensor, state=None, info={}):

        num_nodes = self.gridsize * self.gridsize

        batch_size = observations["graph_nodes"].shape[0]
        data_list = [Data(x=observations["graph_nodes"][index],
                          edge_index=observations["graph_edge_links"][index])
                     for index in range(batch_size)]
        action_masks = observations['mask'].bool()
        batch_data = Batch.from_data_list(data_list)

        x = batch_data.x.float()
        edge_index = batch_data.edge_index.long()
        batch = batch_data.batch

        outs = [x]

        # GIN layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            outs.append(x)

        x = torch.cat(outs, dim=1)

        if self.mode == 'actor':

            if not self.training:
                x = x[action_masks.flatten()]

            x = x.reshape(-1, self.mid_dim)

            x = self.lin1(x)
            x = self.batch_norm(x)
            x = self.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin2(x)

            if not self.training:
                out = torch.full((batch_size * num_nodes, 1), self.min_val)
                out[action_masks.flatten()] = x
                x = out.reshape(-1, num_nodes)
            else:
                x = x.reshape(-1, num_nodes)
                x[~action_masks] = self.min_val
            x = self.softmax(x)
            return x, state

        elif self.mode == 'critic':

            x = self.lin1(x)
            x = self.batch_norm(x)
            x = self.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin2(x)

            x = global_mean_pool(x, batch)

            return x


class EAGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, gridsize: int, n_agents: int, dist_matrix: torch.Tensor,
                 n_graph_layers: int, mode: str, min_val: float):

        super().__init__()

        self.gridsize = gridsize
        self.n_agents = n_agents
        self.num_features = input_dim

        self.dist_matrix = dist_matrix

        # Actor or critic net
        self.mode = mode

        self.n_layers = n_graph_layers

        self.min_val = min_val

        self.convs = torch.nn.ModuleList()

        for _ in range(n_graph_layers):
            mlp = Sequential(
                Linear(input_dim, hidden_dim),
                LayerNorm(hidden_dim),
                ReLU()
            )
            # fully connected graph contains self-loops:
            # eps=-1 avoids considering nodes' features twice as they are already in the neighborhood
            self.convs.append(GINConv(mlp, eps=-1).jittable())
            input_dim = hidden_dim

        self.mid_dim = self.num_features + n_graph_layers * hidden_dim

        self.cat_dim = 2 * self.mid_dim + hidden_dim

        self.lin_edges = Linear(1, hidden_dim)

        self.lin_actor1 = Linear(self.cat_dim, self.mid_dim)
        self.batch_norm_actor = BatchNorm1d(self.mid_dim)
        self.relu_actor = ReLU()
        self.lin_actor2 = Linear(self.mid_dim, 1)

        self.lin_critic1 = Linear(self.mid_dim, 2 * hidden_dim)
        self.batch_norm_critic = BatchNorm1d(2 * hidden_dim)
        self.relu_critic = ReLU()
        self.lin_critic2 = Linear(2 * hidden_dim, 1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, observations: torch.Tensor, state=None, info={}):

        num_nodes = self.gridsize * self.gridsize

        locs = observations['agent_locations']
        batch_size = observations["graph_nodes"].shape[0]

        data_list = [Data(x=observations["graph_nodes"][index],
                          edge_index=observations["graph_edge_links"][index])
                     for index in range(batch_size)]
        action_masks = observations['mask'].bool()
        batch_data = Batch.from_data_list(data_list)

        x = batch_data.x.float()
        edge_index = batch_data.edge_index.long()
        batch = batch_data.batch

        outs = [x]

        # GIN layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            outs.append(x)

        x = torch.cat(outs, dim=1)

        if self.mode == 'actor':

            x = x.reshape(-1, num_nodes, self.mid_dim)
            idx = torch.tensor([i for i in range(batch_size)]).unsqueeze(-1)
            a = x[idx, locs].repeat_interleave(num_nodes, dim=1)
            b = x.repeat(1, self.n_agents, 1)
            c = self.dist_matrix[locs].reshape(-1, num_nodes * self.n_agents, 1)
            c = self.lin_edges(c)
            x = torch.cat((a, b, c), dim=2)

            if not self.training:
                x = x[action_masks]

            x = x.reshape(-1, self.cat_dim)

            x = self.lin_actor1(x)
            x = self.batch_norm_actor(x)
            x = self.relu_actor(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin_actor2(x)

            if not self.training:
                out = torch.full((batch_size, self.n_agents * num_nodes, 1), self.min_val)
                out[action_masks] = x
                x = out.reshape(-1, self.n_agents * num_nodes)
            else:
                x = x.reshape(-1, self.n_agents * num_nodes)
                x[~action_masks] = self.min_val
            x = self.softmax(x)
            return x, state

        elif self.mode == 'critic':

            x = x.reshape(-1, self.mid_dim)

            x = self.lin_critic1(x)
            x = self.batch_norm_critic(x)
            x = self.relu_critic(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin_critic2(x)

            x = global_mean_pool(x, batch)

            return x


class MHGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, gridsize: int, dist_matrix: torch.Tensor,
                 n_graph_layers: int, mode: str, min_val: float):

        super().__init__()

        self.gridsize = gridsize
        self.num_features = input_dim

        self.dist_matrix = dist_matrix

        # Actor or critic net
        self.mode = mode

        self.n_layers = n_graph_layers

        self.min_val = min_val

        self.convs = torch.nn.ModuleList()
        self.econvs = torch.nn.ModuleList()

        for _ in range(n_graph_layers):
            mlp = Sequential(
                Linear(input_dim, hidden_dim),
                LayerNorm(hidden_dim),
                ReLU(),
            )
            # fully connected graph contains self-loops:
            # eps=-1 avoids considering nodes' features twice as they are already in the neighborhood
            self.convs.append(GINConv(mlp, eps=-1).jittable())
            input_dim = hidden_dim

        self.mid_dim = self.num_features + n_graph_layers * hidden_dim

        self.cat_dim = self.mid_dim + hidden_dim

        self.origin_lin1 = Linear(self.mid_dim, hidden_dim * 2)
        self.origin_batch_norm = BatchNorm1d(hidden_dim * 2)
        self.origin_relu = ReLU()
        self.origin_lin2 = Linear(hidden_dim * 2, 1)
        self.origin_softmax = torch.nn.Softmax(dim=1)

        self.lin_edges = Linear(1, hidden_dim)
        self.dest_lin1 = Linear(self.cat_dim, hidden_dim * 2)
        self.dest_batch_norm = BatchNorm1d(hidden_dim * 2)
        self.dest_relu = ReLU()
        self.dest_lin2 = Linear(hidden_dim * 2, 1)
        self.dest_softmax = torch.nn.Softmax(dim=1)

        self.lin_critic1 = Linear(self.mid_dim, 2 * hidden_dim)
        self.batch_norm_critic = BatchNorm1d(2 * hidden_dim)
        self.relu_critic = ReLU()
        self.lin_critic2 = Linear(2 * hidden_dim, 1)

    def origin_head(self, x: torch.Tensor, mask: torch.Tensor, num_nodes: int, batch_size: int):

        if not self.training:
            x = x[mask.flatten()]

        x = self.origin_lin1(x)
        x = self.origin_batch_norm(x)
        x = self.origin_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.origin_lin2(x)

        if not self.training:
            out = torch.full((batch_size * num_nodes, 1), self.min_val)
            out[mask.flatten()] = x
            x = out.reshape(-1, num_nodes)
        else:
            x = x.reshape(-1, num_nodes)
            x[~mask] = self.min_val

        x = self.origin_softmax(x)

        return x

    def destination_head(self, x: torch.Tensor, origins: torch.Tensor, mask: torch.Tensor, num_nodes: int, batch_size: int):

        d = self.dist_matrix[origins].reshape(batch_size * num_nodes, 1)
        d = self.lin_edges(d)

        x = torch.cat((x, d), dim=1)

        if not self.training:
            x = x[mask.flatten()]

        x = self.dest_lin1(x)
        x = self.dest_batch_norm(x)
        x = self.dest_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.dest_lin2(x)

        if not self.training:
            out = torch.full((batch_size * num_nodes, 1), self.min_val)
            out[mask.flatten()] = x
            x = out.reshape(-1, num_nodes)
        else:
            x = x.reshape(-1, num_nodes)
            x[~mask] = self.min_val

        x = self.dest_softmax(x)

        return x

    def forward(self, observations: torch.Tensor, state=None, info={}):

        num_nodes = self.gridsize * self.gridsize

        batch_size = observations["graph_nodes"].shape[0]

        data_list = [Data(x=observations["graph_nodes"][index],
                          edge_index=observations["graph_edge_links"][index])
                     for index in range(batch_size)]
        action_masks = observations['mask'].bool()
        batch_data = Batch.from_data_list(data_list)

        x = batch_data.x.float()
        edge_index = batch_data.edge_index.long()
        batch = batch_data.batch

        outs = [x]

        # GIN layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            outs.append(x)

        x = torch.cat(outs, dim=1)

        if self.mode == 'actor':

            origin_outputs = self.origin_head(x, action_masks[:, 0], num_nodes, batch_size)

            if self.training:
                if 'learn' in info:
                    origins = state[:, 0].long()
                else:
                    m = Categorical(origin_outputs)
                    origins = m.sample()
            else:
                origins = origin_outputs.argmax(dim=1)

            idx = torch.tensor([i for i in range(batch_size)]).unsqueeze(-1)
            action_masks[idx, 1, origins.unsqueeze(-1)] = True

            destination_outputs = self.destination_head(x, origins, action_masks[:, 1], num_nodes, batch_size)

            x = torch.cat((origin_outputs.unsqueeze(dim=1), destination_outputs.unsqueeze(dim=1)), dim=1)

            return x, origins

        elif self.mode == 'critic':

            x = x.reshape(-1, self.mid_dim)

            x = self.lin_critic1(x)
            x = self.batch_norm_critic(x)
            x = self.relu_critic(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin_critic2(x)

            x = global_mean_pool(x, batch)

            return x
