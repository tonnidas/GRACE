## Dataset: Cora (Epochs: 200)
- GRACE:                                                    F1Mi: 0.8370, F1Ma: 0.8259
- GCA (degree):                                             F1Mi: 0.8150, F1Ma: 0.7991
- 3-hop path counts (hyper-p: 1.0, cut-off-p: 1.0):         F1Mi: 0.8355, F1Ma: 0.8244
- SoftMax                                                   F1Mi: 0.8358, F1Ma: 0.8267
- SoftMax (with feature drop)                               F1Mi: 0.8381, F1Ma: 0.8279

## Dataset: CiteSeer (Epochs: 200)
- GRACE:                                                    F1Mi: 0.7029, F1Ma: 0.6503
- GCA (degree):                                             F1Mi: 0.7121, F1Ma: 0.6585
- 3-hop path counts (hyper-p: 1.0, cut-off-p: 1.0):         F1Mi: 0.7017, F1Ma: 0.6544
- SoftMax                                                   F1Mi: 0.7117, F1Ma: 0.6535

## Dataset: Amazon-Photo (Epochs: 2000)
- GRACE:                                                    F1Mi: 0.9252, F1Ma: 0.910
- GCA (degree):                                             F1Mi: 0.9249, F1Ma: 0.9115
- 3-hop path counts (hyper-p: 1.0, cut-off-p: 1.0):         F1Mi: 0.9264, F1Ma: 0.9112
- SoftMax                                                   F1Mi: 0.9262, F1Ma: 0.9110

## Dataset: Amazon-Computers (Epochs: 2000)
- GRACE:                                                    F1Mi: 0.8847, F1Ma: 0.8686
- GCA (degree):                                             F1Mi: 0.8885, F1Ma: 0.8757
- 3-hop path counts (hyper-p: 1.0, cut-off-p: 1.0):         Out of memory in Kodiak
- SoftMax                                                   F1Mi: 0.8848, F1Ma: 0.8684

## Dataset: Coauthor-CS (Epochs: 1000)
- GRACE:                                                    F1Mi: 0.9269, F1Ma: 0.9076
- GCA (degree):                                             F1Mi: 0.9278, F1Ma: 0.9089
- 3-hop path counts (hyper-p: 1.0, cut-off-p: 1.0):         F1Mi: 0.9276, F1Ma: 0.9089
- SoftMax                                                   F1Mi: 0.9284, F1Ma: 0.9092

## Dataset: Coauthor-Phy (Epochs: 1500): Out of memory in Kodiak


## Temp

$ python train.py --dataset Cora --drop_scheme partition --local_weight 0.2 --global_weight 0.1

args: Namespace(dataset='Cora', config='config.yaml', drop_scheme='partition', local_weight=0.2, global_weight=0.1, runs=1)
config: {'seed': 39788, 'learning_rate': 0.0005, 'num_hidden': 128, 'num_proj_hidden': 128, 'activation': 'relu', 'base_model': 'GCNConv', 'num_layers': 2, 'drop_edge_rate_1': 0.2, 'drop_edge_rate_2': 0.4, 'drop_feature_rate_1': 0.3, 'drop_feature_rate_2': 0.4, 'tau': 0.4, 'num_epochs': 200, 'weight_decay': 1e-05}
Running on device: cpu
Number of communities: 101
Run: 0
Epochs: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:29<00:00,  6.69it/s]
F1Mi: 0.8384±0.0046, F1Ma: 0.8246±0.0043
Total runs: 1, avg F1Mi: 0.8384, avg F1Ma: 0.8246

$ python train.py --dataset Cora --drop_scheme partition --local_weight 0.2 --global_weight 0.1 --separate_encoder

args: Namespace(dataset='Cora', config='config.yaml', drop_scheme='partition', local_weight=0.2, global_weight=0.1, separate_encoder=True, runs=1)
config: {'seed': 39788, 'learning_rate': 0.0005, 'num_hidden': 128, 'num_proj_hidden': 128, 'activation': 'relu', 'base_model': 'GCNConv', 'num_layers': 2, 'drop_edge_rate_1': 0.2, 'drop_edge_rate_2': 0.4, 'drop_feature_rate_1': 0.3, 'drop_feature_rate_2': 0.4, 'tau': 0.4, 'num_epochs': 200, 'weight_decay': 1e-05}
Running on device: cpu
Number of communities: 106
Run: 0
Epochs: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:31<00:00,  6.26it/s]
F1Mi: 0.7642±0.0095, F1Ma: 0.7202±0.0136
Total runs: 1, avg F1Mi: 0.7642, avg F1Ma: 0.7202