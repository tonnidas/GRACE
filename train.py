import argparse
import os.path as osp
import random
import yaml
from yaml import SafeLoader
from tqdm import tqdm

import torch
from torch_geometric.utils import dropout_adj

from model import Encoder, Model
from eval import label_classification
from gca import get_drop_weights, get_feature_weights
from dataset import get_dataset
from utils import get_activation, get_base_model, drop_edge_weighted, drop_feature_weighted, drop_feature


def _drop_edge(edge_index, drop_edge_rate):
    if args.drop_scheme in ['degree', 'evc', 'pr']:
        return drop_edge_weighted(edge_index, gca_drop_weights, drop_edge_rate, threshold=0.7)
    else:
        return dropout_adj(edge_index, p=drop_edge_rate)[0]


def _drop_feature(x, drop_feature_rate):
    if args.drop_scheme in ['degree', 'evc', 'pr']:
        return drop_feature_weighted(x, gca_feature_weights, drop_feature_rate)
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


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)
    label_classification(z, y, ratio=0.1)


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

    # ============================================================================
    if args.drop_scheme in ['degree', 'evc', 'pr']:
        gca_drop_weights = get_drop_weights(data, args.drop_scheme).to(device)
        gca_feature_weights = get_feature_weights(
            args.dataset, data, args.drop_scheme).to(device)

    # ============================================================================

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss = train(model, data.x, data.edge_index)
        # if (epoch+1) % 100 == 0:
        #     print('Epoch:', epoch)
        #     test(model, data.x, data.edge_index, data.y, final=False)

    print("=== Final ===")
    test(model, data.x, data.edge_index, data.y, final=True)
