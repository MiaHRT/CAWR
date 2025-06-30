# CAWR

## Introduction

This project is the code for the CAWR (Corruption-averse Advantage-Weighted Regression) algorithm. It provides diverse policy loss functions (i.e. L2, L1, Huber, Skew, Flat) and priorities (i.e. None, Normal, Standard, AW, ODPR, Quantile) for policy optimization and prioritized decoupled resampling. Specific definitions are provided in the paper "CAWR: CORRUPTION-AVERSE ADVANTAGE-WEIGHTED REGRESSION FOR ROBUST POLICY OPTIMIZATION".

This project supports D4RL benchmark Mujoco locomotion tasks: Hopper, Walker2D, and HalfCheetah. We provide demo configs for replicating the paper results and for further use. The demos for pretrain (ablation study) are in the folder named pretrain, with the advPER and no_advPER indicating the codes for using prioritized resampling and not using, respectively. Similarly, the demos for direct training (comparison on the D4RL benchmarks) are in the folder named no_pretrain, with the advPER and no_advPER indicating the codes for using prioritized decoupled resampling and not using, respectively.

Notice that this code is modified based on the DI-engine, you can go to https://github.com/opendilab/DI-engine/tree/main for further information.

The experiments in the paper are conducted using GTX1080Ti, if you use a different GPU, you may not be able to replicate our results precisely because of the random number generation.

## Configuration Instructions

The project is in Python language, please make sure the following libraries are installed before running:

```bash
python==3.9.19
torch==2.5.0
mujoco==3.2.4
mujoco-py==2.1.2.14
D4RL==1.1
DI-engine==0.5.2
```

## Quick Start

To replicate the ablation study (using pre-trained advantage function) for CAWR with $L_1$ norm, you can go to pretrain/no_advPER/ and run:

```bash
python d4rl_pretrain_no_advPER_main.py --seed 10 --config hopper_medium_cawr_L1_config.py
```

To replicate the ablation study (using pre-trained advantage function) for CAWR with $L_1$ norm and Normal PER, you can go to pretrain/advPER/ and run:

```bash
python d4rl_pretrain_advPER_main.py --seed 10 --config hopper_medium_cawr_L1_Normal_config.py
```

To replicate the comparison experiment (not using pre-trained advantage function) for CAWR with $L_2$ norm evaluated on Walker2d-m-v2, you can go to no_pretrain/no_advPER/ and run:

```bash
python d4rl_no_advPER_main.py --seed 10 --config walker2d_medium_cawr_L2_config.py
```

To replicate the comparison experiment (not using pre-trained advantage function) for CAWR with $L_2$ norm and Normal PER evaluated on Walker2d-m-v2, you can go to no_pretrain/advPER/ and run:

```bash
python d4rl_advPER_main.py --seed 10 --config walker2d_medium_cawr_L2_Normal_config.py
```

## Citation

```latex
@misc{cawr,
      title={CAWR: Corruption-Averse Advantage-Weighted Regression for Robust Policy Optimization},
      author={Ranting Hu},
      year={2025},
      eprint={2506.15654},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.15654},
}
```

## License

CAWR released under the Apache 2.0 license.
