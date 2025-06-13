import argparse
import os.path as osp
import random
import yaml
import numpy as np
from yaml import SafeLoader
from tqdm import tqdm

import torch
from torch_geometric.utils import dropout_edge

from model import Encoder, Model
from eval import label_classification
import gca
import experiment
from dataset import get_dataset
from utils import get_activation, get_base_model, drop_edge_weighted, drop_feature_weighted, drop_feature


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


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = _drop_edge(edge_index, drop_edge_rate_1)
    edge_index_2 = _drop_edge(edge_index, drop_edge_rate_2)

    x_1 = _drop_feature(x, drop_feature_rate_1)
    x_2 = _drop_feature(x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y):
    model.eval()
    z = model(x, edge_index)
    return label_classification(z, y, ratio=0.1)


def get_drop_probs():
    if args.drop_scheme in ['degree', 'evc', 'pr']:
        drop_edge_probs = gca.get_drop_weights(data, args.drop_scheme)
        drop_feature_probs = gca.get_feature_weights(args.dataset, data, args.drop_scheme)
    elif args.drop_scheme == 'hop':
        drop_edge_probs = experiment.hop_drop_edge_probs(data.edge_index)
        drop_feature_probs = None
    elif args.drop_scheme == 'softmax':
        drop_edge_probs = experiment.softmax_drop_edge_probs(data.edge_index)
        drop_feature_probs = experiment.softmax_drop_feature_probs(data.edge_index, data.x)
    else:
        drop_edge_probs, drop_feature_probs = None, None

    return drop_edge_probs, drop_feature_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--drop_scheme', type=str, default='uniform')
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

    encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    results_f1mi = []
    results_f1ma = []
    for run in range(5):
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            loss = train(model, data.x, data.edge_index)

        result = test(model, data.x, data.edge_index, data.y)
        f1mi, f1ma = result['F1Mi'], result['F1Ma']
        print(f"Run: {run}, F1Mi: {f1mi['mean']:.4f}+-{f1mi['std']:.4f}, F1Ma: {f1ma['mean']:.4f}+-{f1ma['std']:.4f}")

        results_f1mi.append(f1mi['mean'])
        results_f1ma.append(f1ma['mean'])

    print("=== Final ===")
    print(f"F1Mi: {np.mean(results_f1mi):.4f}, F1Ma: {np.mean(results_f1ma):.4f}")
