import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree
import community as community_louvain

def compute_C(edge_index):
    num_nodes = int(edge_index.max()) + 1

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)

    partition = community_louvain.best_partition(G)
    print('Number of communities:', len(set(partition.values())))

    # Broadcast comparison to get co-membership matrix
    community_labels = torch.tensor([partition[i] for i in range(num_nodes)])
    com_mat = (community_labels[:, None] == community_labels[None, :]).int()

    return com_mat

def compute_B(edge_index, comm_mat):
    num_nodes = int(edge_index.max()) + 1

    degrees = degree(edge_index[0], num_nodes=num_nodes)

    # Top 20% high-degree nodes
    k = max(1, int(0.2 * num_nodes))
    top_nodes = torch.topk(degrees, k=k).indices
    top_nodes_set = set(top_nodes.tolist())

    B = torch.zeros((num_nodes, num_nodes), dtype=torch.int)

    for i in range(num_nodes):
        C_i = set(torch.where(comm_mat[i] == 1)[0].tolist()) # indices where comm_mat[i] == 1
        B_i_indices = top_nodes_set - C_i  # S \ C_i
        B[i, list(B_i_indices)] = 1

    return B
