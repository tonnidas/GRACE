import torch
from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.utils import degree, to_undirected


def get_drop_edge_probs(edge_index, drop_scheme):
    if drop_scheme == 'hop':
        return hop_drop_edge_probs(edge_index)
    elif drop_scheme == 'softmax':
        return softmax_drop_edge_probs(edge_index)
    else:
        return None


def hop_drop_edge_probs(edge_index, k=3, hyper_p=1.0, cut_off_p=1.0):
    print('Calculating hop_drop_edge_probs ...')

    adj = to_torch_coo_tensor(edge_index)

    if k == 2:
        hop_adj = adj @ adj
    elif k == 3:
        hop_adj = adj @ adj @ adj
    else:
        exit('Invalid value of k')

    # Gather the path counts for given edge pairs
    u, v = edge_index[0], edge_index[1]
    path_counts = hop_adj.to_dense()[u, v]

    mx_path_count = max(path_counts)
    avg_path_count = torch.mean(path_counts)

    probs = (mx_path_count - path_counts) / (mx_path_count - avg_path_count)
    probs = torch.clamp(probs * hyper_p, max=cut_off_p)

    # Normalize probs to sum 1
    return probs / sum(probs)


def softmax_drop_edge_probs(edge_index):
    print('Calculating softmax_drop_edge_probs ...')

    node_deg = degree(to_undirected(edge_index)[1])

    normalized_deg = node_deg / max(node_deg)
    exp_deg = torch.exp(normalized_deg)

    adj = to_torch_coo_tensor(edge_index)
    sum_neighbor_exp_deg = adj @ exp_deg

    u, v = edge_index[0], edge_index[1]
    return 1 - (exp_deg[u] + exp_deg[v]) / (sum_neighbor_exp_deg[u] + sum_neighbor_exp_deg[v])
