# GRACE

This repository is adopted from [GRACE](https://github.com/CRIPAC-DIG/GRACE) and [GCA](https://github.com/CRIPAC-DIG/GCA).

```
$ conda create -n grace python=3.11
$ conda activate grace

$ pip install PyYAML tqdm scikit-learn ogb
$ pip install torch torch_geometric torch-scatter
$ conda install pytorch pytorch-scatter pyg -c pyg (if pip does not work)

$ python train.py --dataset Cora --drop_scheme hop
```
