## Overview
This project contains the code for "Adapt to Adapt: A tempo-control Framework for Non-stationary Reinforcement Learning"
The code is separated into two folders. [PTM](./PTM) folder contains algorithm PTM-G and MBPO, [baselines](./baselines) folder contains ProOLS, FTML, ONPG.  
## Installation
> conda env create -f py38_ptm.yml
> 
## Dependencies
Pytorch 1.11.0 & MuJoCo 2.0

## Changes on Mujoco [env].py files for non-statinoary
To make the environment non-stationary, we added some codes on a original mujoco environment files to yield three components of a reward (healthy_reward, forward_reward, ctrl_cost).

After install Mujoco, please replace "swimmer.py", "half_cheetah.py", "hopper.py" files in "gym" site-package (ex: /anaconda3/envs/py38_mbpo/lib/python3.8/site-packages/gym/envs/mujoco )
with corresponding files in a folder [replace_files](./replace_files).
We have highlight the codes that should be added as comment "######## add the following code ########">








## How to run

Train ProOLS, ONPG, FTML (see [execute files](./baselines/Src/exp_swimmer)).
> python run_NS.py --algo_name OFPG --env_name "Swimmer-v2" --speed 1 --actor_lr 1e-3

Train PTM-G (see [execute files](./PTM/exp_swimmer)).
> python main_mbpo_new.py --env_name "Swimmer-v2" --exp_folder_name "swimmer" --noisebound_ns 0.01 --num_train_repeat 50 --use_fbpo True --num_epoch 150 --policyevalNupdateIterNum 1 --non_stationary_reward_setting True --speed 1 --nonstationary_type "r_f_change" --nonstationary_function "sin" --rollout_max_length 3 --get_model_prediction_error True --lr 0.0003;

Train MBPO (see [execute files](./PTM/exp_swimmer)).
> python main_mbpo_new.py --env_name "Swimmer-v2" --exp_folder_name "swimmer" --num_epoch 150 --num_train_repeat 50 --policyevalNupdateIterNum 1 --non_stationary_reward_setting True --speed 1 --nonstationary_type "r_f_change" --nonstationary_function "sin" --rollout_max_length 3 --get_model_prediction_error True --lr 0.0003;

## Reference
The code is build upon the following open source codes:
* The code from the paper "Yash Chandak et al. (2020) Optimizing for the Future in Non-Stationary MDPs": https://github.com/yashchandak/OptFuture_NSMDP .
* The code from the paper "Michal Janner et al. (2019) When to Trust Your Model: Model-Based Policy Optimization": https://github.com/jannerm/mbpo .
* The re-implementation pytorch version of the paper "Michal Janner et al. (2019)": https://github.com/Xingyu-Lin/mbpo_pytorch .
