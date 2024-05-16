# Shared Experience Actor Critic

This repository is the implementation of different robust-improving methods on [Shared Experience Actor Critic](https://arxiv.org/abs/2006.07169). 

## Requirements

For the experiments in LBF and RWARE, please install from:
- [Multi-Robot Warehouse Official Repo](https://github.com/uoe-agents/robotic-warehouse)

Also requires, PyTorch 1.6+

## Training
To train the agents in RWARE, navigate to the seac directory:
```
cd seac
```

And run:

```train
python train.py with <env config>
```

Valid environment configs are: 
- `env_name=rware-tiny-2ag-v1 time_limit=500` 
- `env_name=rware-tiny-4ag-v1 time_limit=500` 
- ...
- `env_name=rware-tiny-2ag-hard-v1 time_limit=500` or any other rware environment size/configuration.

## Evaluation/Visualization

To load and render the pretrained models in SEAC, run in the seac directory

```eval
python evaluate.py
```