for ns in True
do
  for speed in 1 2 3 4 5
  do
    echo $ns
    echo $speed
      CUDA_VISIBLE_DEVICES=0 python ./../../main_mbpo_new.py --env_name "Swimmer-v2" --exp_folder_name "swimmer" --noisebound_ns 0.05 --num_train_repeat 38 --use_fbpo True --num_epoch 300 --policyevalNupdateIterNum 1 --non_stationary_reward_setting $ns --speed $speed --nonstationary_type "r_f_change" --nonstationary_function "sin" --rollout_max_length 3 --get_model_prediction_error True --lr 0.0003;
  done
done



