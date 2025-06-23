# GRACE

This repository is adopted from [GRACE](https://github.com/CRIPAC-DIG/GRACE) and [GCA](https://github.com/CRIPAC-DIG/GCA).

```
$ conda create -n grace python=3.11
$ conda activate grace

$ pip install PyYAML tqdm scikit-learn ogb
$ pip install torch torch_geometric torch-scatter

$ python train.py --dataset Cora --drop_scheme hop
```

## Run in Baylor Kodiak

### Submit a job

```
./kodiak.sh submit {job_name} "{python_args}"

Example: 
$./kodiak.sh submit cora_local_global_sim "--dataset Cora --runs 3 --local_weight 0.2 --global_weight 0.1"
```

### Get job status and download output files

```
$ ./kodiak.sh sync
```
