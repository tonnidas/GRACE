args: Namespace(dataset='Cora', config='config.yaml', drop_scheme='uniform', local_weight=0.3, global_weight=0, runs=3)
config: {'seed': 39788, 'learning_rate': 0.0005, 'num_hidden': 128, 'num_proj_hidden': 128, 'activation': 'relu', 'base_model': 'GCNConv', 'num_layers': 2, 'drop_edge_rate_1': 0.2, 'drop_edge_rate_2': 0.4, 'drop_feature_rate_1': 0.3, 'drop_feature_rate_2': 0.4, 'tau': 0.4, 'num_epochs': 200, 'weight_decay': 1e-05}
Running on device: cuda
Number of communities: 103

Best F1Mi: 0.8324 with weights: Pairwise=0.70, Local=0.20, Global=0.10
Start: 2025-06-19 04:46:58 CDT, End: 2025-06-19 05:06:15 CDT, Duration: 0h:19m:17s

===

args: Namespace(dataset='CiteSeer', config='config.yaml', drop_scheme='uniform', local_weight=0.3, global_weight=0, runs=3)
config: {'seed': 38108, 'learning_rate': 0.001, 'num_hidden': 256, 'num_proj_hidden': 256, 'activation': 'prelu', 'base_model': 'GCNConv', 'num_layers': 2, 'drop_edge_rate_1': 0.2, 'drop_edge_rate_2': 0.0, 'drop_feature_rate_1': 0.3, 'drop_feature_rate_2': 0.2, 'tau': 0.9, 'num_epochs': 200, 'weight_decay': 1e-05}
Running on device: cuda
Number of communities: 470

Best F1Mi: 0.6861 with weights: Pairwise=0.80, Local=0.00, Global=0.20
Start: 2025-06-19 04:49:19 CDT, End: 2025-06-19 05:24:16 CDT, Duration: 0h:34m:57s

===

args: Namespace(dataset='Amazon-Photo', config='config.yaml', drop_scheme='uniform', local_weight=0.3, global_weight=0, runs=3)
config: {'seed': 39788, 'learning_rate': 0.1, 'num_hidden': 256, 'num_proj_hidden': 64, 'activation': 'relu', 'base_model': 'GCNConv', 'num_layers': 2, 'drop_edge_rate_1': 0.3, 'drop_edge_rate_2': 0.5, 'drop_feature_rate_1': 0.1, 'drop_feature_rate_2': 0.1, 'tau': 0.3, 'num_epochs': 2000, 'weight_decay': 1e-05}
Running on device: cuda
Number of communities: 150

Best F1Mi: 0.9247 with weights: Pairwise=0.70, Local=0.20, Global=0.10
Start: 2025-06-19 04:49:31 CDT, End: 2025-06-20 01:21:09 CDT, Duration: 20h:31m:38s

===

args: Namespace(dataset='Amazon-Computers', config='config.yaml', drop_scheme='uniform', local_weight=0.3, global_weight=0, runs=3)
config: {'seed': 39788, 'learning_rate': 0.01, 'num_hidden': 128, 'num_proj_hidden': 128, 'activation': 'rrelu', 'base_model': 'GCNConv', 'num_layers': 2, 'drop_edge_rate_1': 0.6, 'drop_edge_rate_2': 0.3, 'drop_feature_rate_1': 0.2, 'drop_feature_rate_2': 0.3, 'tau': 0.2, 'num_epochs': 2000, 'weight_decay': 1e-05}
Running on device: cuda
Number of communities: 333

Best F1Mi: 0.8821 with weights: Pairwise=0.30, Local=0.60, Global=0.10
Start: 2025-06-19 04:49:40 CDT, End: 2025-06-21 13:25:21 CDT, Duration: 56h:35m:41s

===

args: Namespace(dataset='Coauthor-CS', config='config.yaml', drop_scheme='uniform', local_weight=0.3, global_weight=0, runs=3)
config: {'seed': 39788, 'learning_rate': 0.0005, 'num_hidden': 256, 'num_proj_hidden': 256, 'activation': 'rrelu', 'base_model': 'GCNConv', 'num_layers': 2, 'drop_edge_rate_1': 0.3, 'drop_edge_rate_2': 0.2, 'drop_feature_rate_1': 0.3, 'drop_feature_rate_2': 0.4, 'tau': 0.4, 'num_epochs': 1000, 'weight_decay': 1e-05}
Running on device: cuda
Number of communities: 29

Best F1Mi: 0.9306 with weights: Pairwise=1.00, Local=0.00, Global=0.00
Start: 2025-06-19 04:50:01 CDT, End: 2025-06-21 14:19:50 CDT, Duration: 57h:29m:50s
