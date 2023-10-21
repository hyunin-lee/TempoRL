import argparse
import time
import gym
import torch
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import logging

import os
import os.path as osp
import json

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler

from matplotlib import animation

from torch.utils.tensorboard import SummaryWriter
from pyFTS.data import TAIEX, NASDAQ, SP500, DowJones, Ethereum, Bitcoin, EURGBP, EURUSD, GBPUSD
#from tf_models.constructor import construct_model, format_samples_for_training


def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)') #Ant-v2 #HalfCheetah-v2 #Hopper-v2 #Walker2d-v2 #Humanoid-v2 #Swimmer-v2
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=100, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
                        help='rollout max length')
    parser.add_argument('--num_epoch', type=int, default=100, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.01, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=50, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=4096, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--max_path_length', type=int, default=1000, metavar='A',
                        help='max length of path')


    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')


    parser.add_argument('--exp_folder_name', default="default",
                        help='set non_stationary reward setting')
    parser.add_argument('--non_stationary_reward_setting', default=True,
                        help='set non_stationary reward setting')
    parser.add_argument('--speed', default=1, type=int,
                        help='set speed : 1,2,3,4,5')
    parser.add_argument('--nonstationary_type', default="r_f_change", type=str,
                        help='[1]r_f_change [2]v_d_change')
    parser.add_argument('--nonstationary_function', default="sin", type=str,
                        help='[1]sin [2]square')
    parser.add_argument('--use_fbpo', default=False, type=bool,
                        help='use_fbpo')
    parser.add_argument('--noisebound_ns', type=float, default=0.01,
                        help='noisebound_ns')
    parser.add_argument('--policyevalNupdateIterNum', type=int, default = 1)
    
    parser.add_argument('--past_time_length', type=int, default = 100)

    parser.add_argument('--forecaster_type', type=str, default="arima", metavar='A',
                        help='arima, simple_average')
    parser.add_argument('--sliding_windew_length', type=int, default= 10 , metavar='A',
                        help='sliding windoew for simple average')
    parser.add_argument('--get_model_prediction_error', type=bool, default=False)
    parser.add_argument('--arima_manual_pdq', type=bool, default=False)
    parser.add_argument('--arima_p', type=int, default=-1)
    parser.add_argument('--arima_d', type=int, default=-1)
    parser.add_argument('--arima_q', type=int, default=-1)


    return parser.parse_args()


def return_ns(args,ns_timestep) :
    if args.nonstationary_function == "sin" :
        ns = np.sin(2 * np.pi * args.speed * ns_timestep / 37)
    elif args.nonstationary_function == "bigsin" : 
        ns = 5*np.sin(2 * np.pi * args.speed * ns_timestep / 37) + 2.5
    elif args.nonstationary_function == "square" :
        if np.sin(2 * np.pi * args.speed* ns_timestep / 19) > 0 :
            ns = 1
        else :
            ns = -1
    elif args.nonstationary_function == "real_data" :
        start_point = args.speed * 500 + 500
        past_length = args.past_time_length
        dji = DowJones.get_data('AVG')[1000:5000]
        data = dji[start_point - past_length:start_point + args.num_epoch+1]
        ## resize the data ## 
        data = (2/(np.max(data)-np.min(data))) * (data - np.min(data)) - 1
        ns = data[past_length + ns_timestep]
    elif  args.nonstationary_function in ["ads1","ads2","ads4","lineartrend1","lineartrend2","lineartrend3"] :
        start_point = args.speed * 1 + 100
        past_length = args.past_time_length
        data = np.load("../../dataset/"+args.nonstationary_function+".npy")
        ns = data[past_length + ns_timestep]
    else:
        raise NotImplemented

    if ns == 0 :
        ns = 0.01
        noisy_ns = 0.01
    else :
        noisy_ns = np.random.uniform(ns-args.noisebound_ns,ns+args.noisebound_ns)
    return ns, noisy_ns

def save_frames_as_gif(frames, path, filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='pillow', fps=60)


def return_ns_reward_from_env(args,env_sampler,_non_stationary_timestep) :

    if args.nonstationary_type == "r_f_change":
        ns_reward_fromEnv = env_sampler.env.env.healthy_reward + return_ns(args, _non_stationary_timestep)[
            0] * env_sampler.env.env.forward_reward - env_sampler.env.env.control_cost
        reward_list = [env_sampler.env.env.healthy_reward, return_ns(args, _non_stationary_timestep)[
            0] * env_sampler.env.env.forward_reward, -env_sampler.env.env.control_cost]
    elif args.nonstationary_type == "v_d_change":
        ns_reward_fromEnv = env_sampler.env.env.healthy_reward - np.abs(return_ns(args, _non_stationary_timestep)[
                                                                            0] - env_sampler.env.env.forward_reward) - env_sampler.env.env.control_cost
        reward_list = [env_sampler.env.env.healthy_reward, -np.abs(return_ns(args, _non_stationary_timestep)[
                                                                            0] - env_sampler.env.env.forward_reward),- env_sampler.env.env.control_cost]
    else:
        raise NotImplementedError
    assert len(reward_list) == 3
    return ns_reward_fromEnv , reward_list


def train(args, env_sampler, predict_env, agent, env_pool, model_pool):
    ########################################################
    if args.exp_folder_name == "default" :
        raise NotImplementedError("set the experiment folder name")
    elif args.exp_folder_name == "test" : 
        folder_path = "./../../" + str(args.exp_folder_name) + "/env_name_"+str(args.env_name)+"/speed_" + str(args.speed) + "_nsType_"+str(args.nonstationary_type)+"_nsFunc_"+str(args.nonstationary_function)+"_noisebound_"+str(args.noisebound_ns)+ "_policyEvalUpdateIterNum_"+ str(args.policyevalNupdateIterNum)+"_trainRepeat_"+ str(args.num_train_repeat)+"_initExpSteps_"+str(args.init_exploration_steps)+"_lr_"+str(args.lr)+"_EminEmaxLminLmax_"+str(args.rollout_min_epoch)+str(args.rollout_max_epoch)+str(args.rollout_min_length)+str(args.rollout_max_length)+"_fType_"+str(args.forecaster_type)+"_swl_"+str(args.sliding_windew_length)+"_arima_manual_"+str(args.arima_manual_pdq)+"_pdq_"+str(args.arima_p)+str(args.arima_d)+str(args.arima_q) +"_useFBPO_" + str(args.use_fbpo)
        gif_folder_path = "./../../" + str(args.exp_folder_name) + "/env_name_"+str(args.env_name)+"/speed_" + str(args.speed) + "_nsType_"+str(args.nonstationary_type)+"_nsFunc_"+str(args.nonstationary_function)+"_noisebound_"+str(args.noisebound_ns)+ "_policyEvalUpdateIterNum_"+ str(args.policyevalNupdateIterNum)+"_trainRepeat_"+ str(args.num_train_repeat)+"_initExpSteps_"+str(args.init_exploration_steps)+"_lr_"+str(args.lr)+"_EminEmaxLminLmax_"+str(args.rollout_min_epoch)+str(args.rollout_max_epoch)+str(args.rollout_min_length)+str(args.rollout_max_length)+"_fType_"+str(args.forecaster_type)+"_swl_"+str(args.sliding_windew_length)+"_arima_manual_"+str(args.arima_manual_pdq)+"_pdq_"+str(args.arima_p)+str(args.arima_d)+str(args.arima_q) +"_useFBPO_" + str(args.use_fbpo)
    else: 
        folder_path = "./../../" + str(args.exp_folder_name) + "/env_name_"+str(args.env_name)+"/speed_" + str(args.speed) + "_nsType_"+str(args.nonstationary_type)+"_nsFunc_"+str(args.nonstationary_function)+"_noisebound_"+str(args.noisebound_ns)+ "_policyEvalUpdateIterNum_"+ str(args.policyevalNupdateIterNum)+"_trainRepeat_"+ str(args.num_train_repeat)+"_initExpSteps_"+str(args.init_exploration_steps)+"_lr_"+str(args.lr)+"_EminEmaxLminLmax_"+str(args.rollout_min_epoch)+str(args.rollout_max_epoch)+str(args.rollout_min_length)+str(args.rollout_max_length)+"_fType_"+str(args.forecaster_type)+"_swl_"+str(args.sliding_windew_length)+"_arima_manual_"+str(args.arima_manual_pdq)+"_pdq_"+str(args.arima_p)+str(args.arima_d)+str(args.arima_q) +"_useFBPO_" + str(args.use_fbpo)
        gif_folder_path = "/mnt/" + str(args.exp_folder_name) + "/env_name_"+str(args.env_name)+"/speed_" + str(args.speed) + "_nsType_"+str(args.nonstationary_type)+"_nsFunc_"+str(args.nonstationary_function)+"_noisebound_"+str(args.noisebound_ns)+ "_policyEvalUpdateIterNum_"+ str(args.policyevalNupdateIterNum)+"_trainRepeat_"+ str(args.num_train_repeat)+"_initExpSteps_"+str(args.init_exploration_steps)+"_lr_"+str(args.lr)+"_EminEmaxLminLmax_"+str(args.rollout_min_epoch)+str(args.rollout_max_epoch)+str(args.rollout_min_length)+str(args.rollout_max_length)+"_fType_"+str(args.forecaster_type)+"_swl_"+str(args.sliding_windew_length)+"_arima_manual_"+str(args.arima_manual_pdq)+"_pdq_"+str(args.arima_p)+str(args.arima_d)+str(args.arima_q) +"_useFBPO_" + str(args.use_fbpo)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("The new directory is created!")
#    if not os.path.exists(gif_folder_path):
#        os.makedirs(gif_folder_path)
#        print("The new GIF directory is created!")    


    ########### save args parse ############
    with open(folder_path+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    ########################################
    writer = SummaryWriter(folder_path)
    #########################################################
    total_step = 0
    total_reward_sum = 0
    rollout_length = 1
    non_stationary_timestep = 0
    print("rollout trajectory at init.")
    exploration_before_start(args, env_sampler, env_pool, agent, non_stationary_timestep=0)
    ############### hyunin add ####################
    Total_step_list = []
    Total_rewardEpoch_list = []
    Total_rewardCum_list = []
    Total_healthyrewardEpoch_list= []
    Total_forwardrewardEpoch_list = []
    Total_controlcostEpoch_list = []
    # Total_EvalrewardEpoch_list = []
    # Total_EvalrewardCum_list = []
    True_non_stationary_var_list = []
    Predict_non_stationary_var_list = []
    frames_list =  []
    model_loss_list = []
    ###############################################


    #### set forecastor ####
    if args.use_fbpo :
        #from forecaster import forecaster
        from forecaster import forecaster_arima, forecaster_simpleaverage, forecaster_arima_manual
        if args.nonstationary_function == "sin"  :
            #f_order = 1
            #f = forecaster_arima(NS_variable_dim=1, ahead_length=1,solvertype="fourier",order=f_order)
            #print("forecastor structure : " + str(f.fitting_solver.model_dict))
            pass
        elif args.nonstationary_function == "bigsin" : 
            pass
        elif args.nonstationary_function == "square" :
            #f_order = 10
            #f = forecaster_arima(NS_variable_dim=1, ahead_length=1,solvertype="fourier",order=f_order)
            #print("forecastor structure : " + str(f.fitting_solver.model_dict))
            pass
        elif args.nonstationary_function == "real_data" :
            #f = forecaster_arima(NS_variable_dim=1)
            pass
        elif args.nonstationary_function in ["ads1","ads2","ads4"] :
            #f = forecaster_arima(NS_variable_dim=1)
            pass
        else :
            raise NotImplemented
        if args.forecaster_type == "arima" : 
            if args.arima_manual_pdq : 
                f = forecaster_arima_manual(NS_variable_dim=1,p=args.arima_p,d=args.arima_d,q=args.arima_q)
            else : 
                f = forecaster_arima(NS_variable_dim=1)
        elif args.forecaster_type == "simple_average" :
            f = forecaster_simpleaverage(NS_variable_dim=1,sliding_windew_length=args.sliding_windew_length)
        else : 
            raise NotImplemented
        past_time_length = args.past_time_length
        past_time_step = np.arange(-past_time_length, 0)
        past_non_stationary_var = [return_ns(args, p_ts)[1] for p_ts in past_time_step]
        for p_ns_var, p_ts in zip(past_non_stationary_var, past_time_step):
            f.update_nonstationary_variable(p_ts, [p_ns_var])



    for epoch_step in range(args.num_epoch):
        print("=================================")
        if epoch_step < args.num_epoch :
            print("epoch_step : "+ str(epoch_step))
            eval_t = False
        else :
            print("EVAL epoch_step : " + str(epoch_step))
            eval_t = True
        start_step = total_step
        train_policy_steps = 0
        reward_sum_thisEpoch = 0
        r_forward_sum_thisEpoch = 0
        r_healthy_sum_thisEpoch = 0
        r_control_sum_thisEpoch = 0
        frames = []
        print("current non-stationary : "+str(non_stationary_timestep))
        for i in count():
            cur_step = total_step - start_step
             
            # frames.append(env_sampler.env.env.render(mode="rgb_array"))

            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                print("last step!")
                break

            cur_state, action, next_state, _, done, info = env_sampler.sample(agent,eval_t=eval_t)
            ns_reward_fromEnv, ns_reward_fromEnv_list = return_ns_reward_from_env(args, env_sampler, non_stationary_timestep)
            noisy_ns_variable = return_ns(args,non_stationary_timestep)[1]
            reward_sum_thisEpoch += ns_reward_fromEnv
            r_healthy_sum_thisEpoch += ns_reward_fromEnv_list[0]
            r_forward_sum_thisEpoch += ns_reward_fromEnv_list[1]
            r_control_sum_thisEpoch += ns_reward_fromEnv_list[2]

            if args.use_fbpo :
                env_pool.push(cur_state, action, ns_reward_fromEnv, next_state, done, noisy_ns_variable)
            else :
                env_pool.push(cur_state, action, ns_reward_fromEnv, next_state, done, None)
            #############################################

            #### model training for future ###
            if cur_step % args.epoch_length == args.epoch_length-1 :
                if args.real_ratio >= 1.0 : 
                    raise ValueError('real ratio should be < 1.0')
                print("/*train model starts")
                train_predict_model(args, env_pool, predict_env)
                print("\*train model ends")
                if args.use_fbpo :
                    print("/*train forecaster starts")
                    f.update_nonstationary_variable(non_stationary_timestep, [return_ns(args,non_stationary_timestep)[1]])
                    f.fit_forecastor()
                    future_ns_predict = f.predict_nonstationary_variable(non_stationary_timestep)
                    future_ns_true = return_ns(args,non_stationary_timestep+1)[0]
                    print("predict: " + str(future_ns_predict) + " , true: " + str(future_ns_true))
                    print("\*train forecaster ends")
                else :
                    future_ns_predict = None
                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)
                mean_model_loss_list2 = []
                ############ policy evaluation and policy update ##################
                for iter_num in range(args.policyevalNupdateIterNum) : 
                    print("loop iteration : "+str(iter_num))
                    print("/*rollout synthetic trajectories with trained model starts")
                    print("non-stationary time step for reward  : " + str(non_stationary_timestep+1))
                    
                    mean_model_loss = rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length, future_ns_predict)
                    model_loss_list.append([total_step, iter_num, mean_model_loss])
                    mean_model_loss_list2.append(mean_model_loss)
                    print("\*rollout synthetic trajectories with trained model ends")
                    ### policy training for later on ##
                    if len(env_pool) > args.min_pool_size:
                        print("/*train policy start")
                        train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)
                        print("\*train policy end")

                #### trash the model_pool data ###
                if args.use_fbpo :
                    print("reset model data")
                    model_pool.reset()
                ##################################


            if cur_step % args.epoch_length == args.epoch_length-1 :
                total_reward_sum +=reward_sum_thisEpoch
                writer.add_scalar("reward/train", reward_sum_thisEpoch, total_step)
                writer.add_scalar("reward_healthy/train,",r_healthy_sum_thisEpoch, total_step)
                writer.add_scalar("reward_forward/train,", r_forward_sum_thisEpoch, total_step)
                writer.add_scalar("reward_control/train,", r_control_sum_thisEpoch, total_step)
                writer.add_scalar("cumulative_reward/train", total_reward_sum, total_step)
                writer.add_scalar("model_mse_loss/train", np.mean(mean_model_loss_list2), total_step)
                Total_step_list.append(total_step)
                Total_rewardEpoch_list.append(total_reward_sum)
                Total_rewardCum_list.append(total_reward_sum)
                Total_healthyrewardEpoch_list.append(r_healthy_sum_thisEpoch)
                Total_forwardrewardEpoch_list.append(r_forward_sum_thisEpoch)
                Total_controlcostEpoch_list.append(r_control_sum_thisEpoch)
                if args.use_fbpo :
                    True_non_stationary_var_list.append(future_ns_true)
                    Predict_non_stationary_var_list.append(future_ns_predict)
                print("total_step : " +str(total_step) + "   |    sum_reward : "+str(total_reward_sum))

                ## save frame ##
                # print("save frame")
                # env_sampler.env.env.close()
                # save_frames_as_gif(frames, path = gif_folder_path, filename = "/"+str(total_step)+".gif")

            total_step += 1
        non_stationary_timestep +=1

        if epoch_step % 50 == 0 : 
            np.save(folder_path + '/total_step.npy', np.array(Total_step_list))
            np.save(folder_path + '/total_rewardEpoch.npy', np.array(Total_rewardEpoch_list))
            np.save(folder_path + '/total_rewardCum.npy', np.array(Total_rewardCum_list))
            np.save(folder_path + '/total_healthyrewardEpoch.npy', np.array(Total_healthyrewardEpoch_list))
            np.save(folder_path + '/total_forwardrewardEpoch.npy', np.array(Total_forwardrewardEpoch_list))
            np.save(folder_path + '/total_controlcostEpoch.npy', np.array(Total_controlcostEpoch_list))
            if args.use_fbpo:
                f.save_model(folder_path)
                np.save(folder_path + '/True_nonstationary_var.npy', np.array(True_non_stationary_var_list))
                np.save(folder_path + '/Predict_nonstationary_var.npy', np.array(Predict_non_stationary_var_list))
                save_plot(y_pred=Predict_non_stationary_var_list,y_true=True_non_stationary_var_list,folder_path=folder_path)
    ## save torch model ##
    torch.save(agent.critic.state_dict(), folder_path+"/critic.pth")
    torch.save(agent.critic_target.state_dict(), folder_path + "/critic_target.pth")
    torch.save(agent.policy.state_dict(), folder_path + "/policy.pth")
    writer.flush()
    writer.close()

def save_plot(y_pred,y_true,folder_path):
    assert len(y_pred) == len(y_true)
    x = [i for i in range(len(y_pred))]
    plt.figure(1)
    plt.plot(x,y_pred,"*-",color="b",label="pred")
    plt.plot(x,y_true,"o-",color="g",label="true")
    plt.xlabel("time step")
    plt.ylabel("ns")
    plt.title("pred vs true")
    plt.savefig(folder_path + "/ns_figure.png")

def exploration_before_start(args, env_sampler, env_pool, agent, non_stationary_timestep):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        ############# hyunin change ################
        # if args.nonstationary_type == "r_f_change":
        #     ns_reward_fromEnv = env_sampler.env.env.healthy_reward + return_ns(args,non_stationary_timestep)[0] * env_sampler.env.env.forward_reward - env_sampler.env.env.control_cost
        # elif args.nonstationary_type == "v_d_change":
        #     ns_reward_fromEnv = env_sampler.env.env.healthy_reward - np.abs(return_ns(args,non_stationary_timestep)[0] - env_sampler.env.env.forward_reward) - env_sampler.env.env.control_cost
        # else:
        #     raise NotImplementedError
        ns_reward_fromEnv, _ = return_ns_reward_from_env(args,env_sampler,non_stationary_timestep)
        noisy_ns_variable = return_ns(args,non_stationary_timestep)[1]

        #############################################
        if args.use_fbpo :
            env_pool.push(cur_state, action, ns_reward_fromEnv, next_state, done, noisy_ns_variable)
        else :
            env_pool.push(cur_state, action, ns_reward_fromEnv, next_state, done, None)
        #############################################

def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    if args.use_fbpo :
        state, action, reward, next_state, done, noisy_ns  = env_pool.sample(len(env_pool))
        noisy_ns = np.expand_dims(noisy_ns,axis=1)
        state = np.concatenate((state,noisy_ns),axis=1)
        next_state = np.concatenate((next_state,noisy_ns),axis=1)
    else :
        state, action, reward, next_state, done, _ = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    if args.use_fbpo :
        assert delta_state[:,-1].all() == 0
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length, future_ns):
    if args.use_fbpo :
        state, action, reward, next_state, done, _ = env_pool.sample_all_batch(args.rollout_batch_size)
    else :
        state, action, reward, next_state, done ,_ = env_pool.sample_all_batch(args.rollout_batch_size)
    print("current model pool size : " + str(model_pool.capacity))
    print("current rollout length  : " + str(rollout_length))
    if args.get_model_prediction_error : 
        model_loss_mse_list = []
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        if args.use_fbpo :
            changed_future_ns = future_ns * np.ones(state.shape[0])
            print("current state length : "+str(changed_future_ns.shape[0]))
            changed_future_ns = np.expand_dims(changed_future_ns, axis=1)
            # print(changed_future_ns.shape)
            # print(state.shape)
            modified_state = np.concatenate((state, changed_future_ns), axis=1)
            next_states, rewards, terminals, info = predict_env.step(modified_state, action)
            next_states = next_states[:,:-1]
        else :
            print("current state length : " + str(state.shape[0]))
            next_states, rewards, terminals, info = predict_env.step(state, action)
        if args.get_model_prediction_error : 
            model_loss_mse = (np.square(next_states - next_state)).mean()
            
            print("model loss : "+str(model_loss_mse))
            model_loss_mse_list.append(model_loss_mse)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j], None) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]
    
    if args.get_model_prediction_error : 
        mean_model_loss_mse = np.mean(model_loss_mse_list)
    else : 
        mean_model_loss_mse = 0

    return mean_model_loss_mse


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    # if total_step % args.train_every_n_steps > 0:
    #     return 0
    #
    # if train_step > args.max_train_repeat_per_step * total_step:
    #     return 0

    for i in range(args.num_train_repeat):
        if i == args.num_train_repeat-1 :
            print("train repeat "+str(i+1))
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size
        ############ hyunin change ###################
        env_state, env_action, env_reward, env_next_state, env_done, _ = env_pool.sample(int(env_batch_size))
        ##############################################
        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done, _ = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                np.concatenate((env_state, model_state), axis=0), \
                np.concatenate((env_action, model_action), axis=0), \
                np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                np.concatenate((env_next_state, model_next_state),axis=0), \
                np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

    return args.num_train_repeat


from gym.spaces import Box


class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs


def main(args=None):
    if args is None:
        args = readParser()

    # Initial environment
    env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    ns_var_size = 1
    ########## hyunin change ############
    if args.use_fbpo :
        state_size += ns_var_size
    ######################
    if args.model_type == 'pytorch':
        env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay)
    else:
        env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args.pred_hidden_size, num_networks=args.num_networks,
                                    num_elites=args.num_elites)

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name, args.model_type)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    # print(args.model_retain_epochs)
    # print(model_steps_per_epoch)
    # print(new_pool_size)
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)
    train(args, env_sampler, predict_env, agent, env_pool, model_pool)


if __name__ == '__main__':
    main()
