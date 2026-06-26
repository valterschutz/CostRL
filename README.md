# Towards Cost Sensitive Decision Making

## Setup

This repository includes a `requirements.txt` verified with `uv` and Python 3.10.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.lock
```

Run commands through the virtualenv Python:

```bash
.venv/bin/python scripts/run_agent.py --cfg_file=configs/paper/sepsis_seque_mbppo_cost_001.yaml --mode=train
```

Notes:

- The original `environment.txt` is a conda export for Python 3.7 and PyTorch 1.11.
- The uv environment uses `torch==2.2.2+cpu` because `torch==1.11.0` failed to load on this system.
- The package is `nflows`, not `nflow`; the old `requirements.txt` name was a typo.
- `requirements.txt` lists direct dependencies; `requirements.lock` pins the full resolved uv environment.

## Usage
```bash
python scripts/run_agent.py --cfg_file=path/to/config --mode=train/test
```

## Code Structure
    .
    |- scripts
    |- src
    |   |- agents
    |   |   |- fully_observed_ppo: fully observed
    |   |   |- concat_action_ppo: concatenate afa and tsk action space
    |   |   |- concat_action_mbppo: with generative model
    |   |   |- batch_hier_ppo: batch acquisition
    |   |   |- batch_hier_mbppo: with generative model
    |   |   |- seque_hier_ppo: sequential acquisition
    |   |   |- seque_hier_mbppo: with generative model
    |   |- environemnts
    |   |   |- sepsis: sepsis simulator
    |   |   |- episode_length_wrapper: limit episode length
    |   |   |- concat_action_wrapper: concat action space
    |   |   |- batch_acquire_env: batch acquisition
    |   |   |- seque_acquire_env: sequential acquisition
    |   |- models
    |   |   |- poex_vae_cat_dis: POEx model for environments with categorical observations and discrete actions
    |   |- networks
    |   |- policies
    |   |- utils
    |- requirements.txt
    |- README.md
