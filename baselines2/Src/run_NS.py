#!~miniconda3/envs/pytorch/bin python
# from __future__ import print_function
import sys
sys.path.insert(0, '../')
import numpy as np
import Src.Utils.utils as utils
from Src.NS_parser import Parser
from Src.config import Config
from time import time
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
from pyFTS.data import TAIEX, NASDAQ, SP500, DowJones, Ethereum, Bitcoin, EURGBP, EURUSD, GBPUSD

class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]

        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.model = config.algo(config=config)

    ##################################################################################################################
    ######################################### hyunin change ##########################################################
    def return_ns(self, ns_timestep):
        if self.config.nonstationary_function == "sin":
            ns = np.sin(2 * np.pi * self.config.speed * ns_timestep / 37)
        elif self.config.nonstationary_function == "square":
            if np.sin(2 * np.pi * self.config.speed * ns_timestep / 19) > 0:
                ns = 1
            else:
                ns = -1
        elif self.config.nonstationary_function == "real_data" :
            start_point = self.config.speed * 500 + 500
            past_length = 100
            dji = DowJones.get_data('AVG')[1000:5000]
            data = dji[start_point - past_length:start_point + self.config.max_episodes+1]
            ## resize the data ##
            data = (2/(np.max(data)-np.min(data))) * (data - np.min(data)) - 1
            ns = data[past_length + ns_timestep]
        else:
            raise NotImplemented

        if ns == 0:
            ns = 0.01
            noisy_ns = 0.01
        else:
            noisy_ns = np.random.uniform(ns - self.config.noisebound_ns, ns + self.config.noisebound_ns)
        return ns, noisy_ns

    def return_ns_reward_from_env(self,_non_stationary_timestep):
        if self.config.nonstationary_type == "r_f_change":
            ns_reward_fromEnv = self.config.env.healthy_reward + self.return_ns(_non_stationary_timestep)[
                0] * self.config.env.forward_reward - self.config.env.control_cost
            reward_list = [self.config.env.healthy_reward, self.return_ns(_non_stationary_timestep)[
                0] * self.config.env.forward_reward, - self.config.env.control_cost]
        elif self.config.nonstationary_type == "v_d_change":
            ns_reward_fromEnv = self.config.env.healthy_reward - np.abs(self.return_ns(_non_stationary_timestep)[
                                                                                0] - self.config.env.forward_reward) - self.config.env.control_cost
            reward_list = [self.config.env.healthy_reward, -np.abs(self.return_ns(_non_stationary_timestep)[
                                                                           0] - self.config.env.forward_reward),
                           - self.config.env.control_cost]
        else:
            raise NotImplementedError
        assert len(reward_list) == 3
        return ns_reward_fromEnv, reward_list
    #######################################################################################################################
    #######################################################################################################################

    def train(self):

        Total_step_list = []
        Total_rewardEpoch_list = []
        Total_rewardCum_list = []
        Total_healthyrewardEpoch_list = []
        Total_forwardrewardEpoch_list = []
        Total_controlcostEpoch_list = []
        True_non_stationary_var_list = []
        if self.config.env_name == "nscartpole_v0" :
            self.config.paths['results'] = self.config.paths['results'] + "/algo_" + str(
                self.config.algo_name) + "_ep_" + str(self.config.max_episodes) + "_speed_" + str(
                self.config.speed) + "_alr_" + str(self.config.actor_lr) + "/"
        else :
            self.config.paths['results'] = self.config.paths['results'] + "/algo_" + str(
                self.config.algo_name) + "_ep_" + str(self.config.max_episodes) + "_speed_" + str(
                self.config.speed) + "_nsType_"+str(self.config.nonstationary_type)+"_nsFunc_"+str(
                self.config.nonstationary_function)+"_noisebound_"+str(self.config.noisebound_ns)+ "_alr_" + str(self.config.actor_lr)
        if not os.path.exists(self.config.paths['results']):
            os.mkdir(self.config.paths['results'])

        self.writer = SummaryWriter(self.config.paths['results'])

        ckpt = 1
        start_ep =  0
        total_steps = 0
        total_reward_sum = 0
        ######## hyunin change ########
        non_stationary_timestep = 0
        ###############################
        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode

            state = self.env.reset()
            self.model.reset()

            step, total_r_thisEpisode = 0, 0
            r_h_thisEpisode, r_f_thisEpisode, r_c_thisEpisode = 0,0,0
            done = False
            while not done:

                action, extra_info, dist = self.model.get_action(state)
                new_state, _, done, info = self.env.step(action=action)
                ##################### hyunin change ###################
                current_ns = self.return_ns(non_stationary_timestep)[0]
                ns_reward, ns_reward_list = self.return_ns_reward_from_env(non_stationary_timestep)
                r_h_thisEpisode += ns_reward_list[0]
                r_f_thisEpisode += ns_reward_list[1]
                r_c_thisEpisode += ns_reward_list[2]
                #######################################################
                if step>=self.config.max_steps-1 :
                    done = True
                self.model.update(state, action, extra_info, ns_reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r_thisEpisode += ns_reward
                # regret += (reward - info['Max'])
                step += 1
                if step >= self.config.max_steps:
                    break

            total_steps += step

            total_reward_sum += total_r_thisEpisode

            self.writer.add_scalar("reward/train", total_r_thisEpisode, total_steps)
            self.writer.add_scalar("cumulative_reward/train", total_reward_sum, total_steps)
            self.writer.add_scalar("reward_healthy/train,", r_h_thisEpisode, total_steps)
            self.writer.add_scalar("reward_forward/train,", r_f_thisEpisode, total_steps)
            self.writer.add_scalar("reward_control/train,", r_c_thisEpisode, total_steps)

            Total_step_list.append(total_steps)
            Total_rewardEpoch_list.append(total_r_thisEpisode)
            Total_rewardCum_list.append(total_reward_sum)
            Total_healthyrewardEpoch_list.append(r_h_thisEpisode)
            Total_forwardrewardEpoch_list.append(r_f_thisEpisode)
            Total_controlcostEpoch_list.append(r_c_thisEpisode)
            True_non_stationary_var_list.append(current_ns)

            t0 = time()
            non_stationary_timestep += 1

            print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                  format(episode, total_r_thisEpisode, total_steps/ckpt, (time() - t0)/ckpt, (time() - t0)/total_steps, self.model.entropy, self.model.get_grads()))



            if episode == self.config.max_episodes -1 :
                np.save(self.config.paths['results'] + '/total_step.npy', np.array(Total_step_list))
                np.save(self.config.paths['results'] + '/total_rewardEpoch.npy', np.array(Total_rewardEpoch_list))
                np.save(self.config.paths['results'] + '/total_rewardCum.npy', np.array(Total_rewardCum_list))
                np.save(self.config.paths['results'] + '/total_healthyrewardEpoch.npy', np.array(Total_healthyrewardEpoch_list))
                np.save(self.config.paths['results'] + '/total_forwardrewardEpoch.npy', np.array(Total_forwardrewardEpoch_list))
                np.save(self.config.paths['results'] + '/total_controlcostEpoch.npy', np.array(Total_controlcostEpoch_list))
                # utils.save_plots(reward_list, config=self.config, name='{seed}_return_history_algo_{algo}_ep{ep}_step{step}_speed{speed}_actorlr{actor_lr}_entropylambda{entropy}_delta{delta}'.format(
                #             seed=self.config.seed,
                #             algo=self.config.algo_name,
                #             ep=self.config.max_episodes,
                #             step=self.config.max_steps,
                #             speed=self.config.speed,
                #             actor_lr="{:e}".format(self.config.actor_lr),
                #             entropy=self.config.entropy_lambda,
                #             delta=self.config.delta
                #         ))
                # utils.save_plots(cum_reward_list, config=self.config, name='{seed}_cum_return_algo_{algo}_ep{ep}_step{step}_speed{speed}_actorlr{actor_lr}_entropylambda{entropy}_delta{delta}'.format(
                #             seed=self.config.seed,
                #             algo=self.config.algo_name,
                #             ep=self.config.max_episodes,
                #             step=self.config.max_steps,
                #             speed=self.config.speed,
                #             actor_lr="{:e}".format(self.config.actor_lr),
                #             entropy=self.config.entropy_lambda,
                #             delta=self.config.delta
                #         ))

        # if self.config.debug and self.config.env_name == 'NS_Reco':
        #
        #     fig1, fig2 = plt.figure(figsize=(8, 6)), plt.figure(figsize=(8, 6))
        #     ax1, ax2 = fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)
        #
        #     action_prob = np.array(action_prob).T
        #     true_rewards = np.array(true_rewards).T
        #
        #     for idx in range(len(dist)):
        #         ax1.plot(action_prob[idx])
        #         ax2.plot(true_rewards[idx])
        #
        #     plt.show()



# @profile
def main(train=True, inc=-1, hyper='default', base=-1):
    t = time()
    args = Parser().get_parser().parse_args()

    # Use only on-policy method for oracle
    if args.oracle >= 0:
            args.algo_name = 'ONPG'

    if inc >= 0 and hyper != 'default' and base >= 0:
        args.inc = inc
        args.hyper = hyper
        args.base = base

    config = Config(args)
    solver = Solver(config=config)

    # Training mode
    if train:
        solver.train()

    print("Total time taken: {}".format(time()-t))

if __name__ == "__main__":
        main(train=True)

