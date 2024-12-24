import torch
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils import degree, to_networkx
from torch_scatter import scatter
import networkx as nx


def get_drop_weights(data, drop_scheme):
    if drop_scheme == 'degree':
        drop_weights = degree_drop_weights(data.edge_index)
    elif drop_scheme == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200)
    elif drop_scheme == 'evc':
        drop_weights = evc_drop_weights(data)
    else:
        drop_weights = None

    return drop_weights


def get_feature_weights(dataset, data, drop_scheme):
    if drop_scheme == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(
                data.x, node_c=node_deg)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg)
    elif drop_scheme == 'pr':
        node_pr = compute_pr(data.edge_index)
        if dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(
                data.x, node_c=node_pr)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr)
    elif drop_scheme == 'evc':
        node_evc = eigenvector_centrality(data)
        if dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(
                data.x, node_c=node_evc)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc)
    else:
        feature_weights = torch.ones((data.x.size(1),))

    return feature_weights


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)
