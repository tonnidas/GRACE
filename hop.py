import torch
import numpy as np
from torch_geometric.utils import remove_self_loops, to_edge_index, to_torch_coo_tensor


def get_drop_edge_probs(edge_index):
    path_counts_3_hop = k_hop_path_counts(edge_index, 3)
    drop_edge_probs_3_hop = drop_edge_probs_for_path_counts(
        path_counts_3_hop, 0.9, 0.3)
    return torch.tensor(drop_edge_probs_3_hop)


def get_hop_edge_index(edge_index):
    print('calculating hop_edge_index, num_edges:', len(edge_index[0]))

    adj = to_torch_coo_tensor(edge_index)
    hop_adj = adj @ adj
    hop_edge_index = to_edge_index(hop_adj)[0]
    hop_edge_index = remove_self_loops(hop_edge_index)[0]

    print('num_hop_edges:', len(hop_edge_index[0]))
    return hop_edge_index


def k_hop_path_counts(edge_index, k):
    num_edges = len(edge_index[0])
    print('calculating k_hop_path_counts, k = {}, num_edges = {}'.format(k, num_edges))

    adj = to_torch_coo_tensor(edge_index)

    if k == 2:
        hop_adj = adj @ adj
    elif k == 3:
        hop_adj = adj @ adj @ adj

    # Gather the path counts for given edge pairs
    u, v = edge_index[0], edge_index[1]
    path_counts = hop_adj.to_dense()[u, v].tolist()

    # path_counts = []
    # for i in range(num_edges):
    #     u, v = int(edge_index[0][i]), int(edge_index[1][i])
    #     path_counts.append(int(hop_adj[u][v]))

    return path_counts


def drop_edge_probs_for_path_counts(path_counts, hyper_p, cut_off_p):
    num_edges = len(path_counts)
    print('calculating drop_edge_prob, num_edges:', num_edges)

    mx = max(path_counts)
    avg = np.mean(path_counts)

    probs = []
    for each in path_counts:
        p = (mx-each)/(mx-avg)
        # p = 1 - min(p * hyper_p, cut_off_p)
        p = min(p * hyper_p, cut_off_p)
        p = round(p, 6)
        probs.append(p)

    # normalize probs to sum 1
    probs = probs / np.sum(probs)

    return probs
