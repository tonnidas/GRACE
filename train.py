import argparse
import os.path as osp
import random
import yaml
import numpy as np
from yaml import SafeLoader
from tqdm import tqdm

import torch
from torch_geometric.utils import dropout_edge, degree

from model import Encoder, Model
from eval import label_classification
import gca
import experiment, precalculation
from dataset import get_dataset
from utils import get_activation, get_base_model, drop_edge_weighted, drop_feature_weighted, drop_feature, grid_search_weights


def _drop_edge(edge_index, drop_edge_rate):
    if drop_edge_probs is not None:
        return drop_edge_weighted(edge_index, drop_edge_probs, drop_edge_rate, threshold=0.7)
    else:
        return dropout_edge(edge_index, p=drop_edge_rate)[0]


def _drop_feature(x, drop_feature_rate):
    if drop_feature_probs is not None:
        return drop_feature_weighted(x, drop_feature_probs, drop_feature_rate)
    else:
        return drop_feature(x, drop_feature_rate)


def train(model: Model, optimizer):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = _drop_edge(data.edge_index, drop_edge_rate_1)
    edge_index_2 = _drop_edge(data.edge_index, drop_edge_rate_2)

    x_1 = _drop_feature(data.x, drop_feature_rate_1)
    x_2 = _drop_feature(data.x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, precalculated, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model):
    model.eval()
    z = model(data.x, data.edge_index)
    return label_classification(z, data.y, ratio=0.1)


def get_drop_probs():
    if args.drop_scheme in ['degree', 'evc', 'pr']: # GCA
        drop_edge_probs = gca.get_drop_weights(data, args.drop_scheme)
        drop_feature_probs = gca.get_feature_weights(args.dataset, data, args.drop_scheme)
    elif args.drop_scheme == 'hop':
        drop_edge_probs = experiment.hop_drop_edge_probs(data.edge_index)
        drop_feature_probs = None
    elif args.drop_scheme == 'softmax':
        drop_edge_probs = experiment.softmax_drop_edge_probs(data.edge_index)
        drop_feature_probs = experiment.softmax_drop_feature_probs(data.edge_index, data.x)
    else: # GRACE
        drop_edge_probs, drop_feature_probs = None, None

    return drop_edge_probs, drop_feature_probs

def single_run():
    # Initialize model and optimizer for each run
    encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss = train(model, optimizer)

    result = test(model)
    f1mi, f1ma = result['F1Mi'], result['F1Ma']
    print(f"F1Mi: {f1mi['mean']:.4f}±{f1mi['std']:.4f}, F1Ma: {f1ma['mean']:.4f}±{f1ma['std']:.4f}")

    return f1mi['mean'], f1ma['mean']

def multiple_runs():
    results_f1mi = []
    results_f1ma = []

    for run in range(args.runs):
        print("Run:", run)
        f1mi, f1ma = single_run()
        results_f1mi.append(f1mi)
        results_f1ma.append(f1ma)
    
    avg_f1mi = np.mean(results_f1mi)
    avg_f1ma = np.mean(results_f1ma)
    print(f"Total runs: {args.runs}, avg F1Mi: {avg_f1mi:.4f}, avg F1Ma: {avg_f1ma:.4f}")
    
    return avg_f1mi, avg_f1ma

def grid_search_run():
    best_f1 = -1
    best_weights = None

    for w0, w1, w2 in grid_search_weights():
        print("Grid serach weights:", w0, w1, w2)
        precalculated["LW"], precalculated["GW"] = w1, w2
        avg_f1mi, avg_f1ma = multiple_runs()

        if avg_f1mi > best_f1:
            best_f1 = avg_f1mi
            best_weights = (w0, w1, w2)
        
        print(f"Best F1Mi: {best_f1:.4f} with weights: Pairwise={best_weights[0]:.2f}, Local={best_weights[1]:.2f}, Global={best_weights[2]:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--drop_scheme', type=str, default='uniform')
    parser.add_argument('--local_weight', type=float, default=0)
    parser.add_argument('--global_weight', type=float, default=0)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    print("args:", args)
    print("config:", config)

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = get_activation(config['activation'])
    base_model = get_base_model(config['base_model'])
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print('Running on device:', device)

    data = data.to(device)

    drop_edge_probs, drop_feature_probs = get_drop_probs()
    if drop_edge_probs is not None:
        drop_edge_probs = drop_edge_probs.to(device)
    if drop_feature_probs is not None:
        drop_feature_probs = drop_feature_probs.to(device)

    if args.local_weight + args.global_weight > 0:
        num_nodes = int(data.edge_index.max()) + 1
        C = precalculation.compute_C(data.edge_index).to(device)
        B = precalculation.compute_B(data.edge_index, C).to(device)
        D = degree(data.edge_index[0], num_nodes=num_nodes).to(device)
        precalculated = {"C": C, "B": B, "D": D, "LW": args.local_weight, "GW": args.global_weight}
    else:
        precalculated = None

    grid_search_run()
