import copy
import numpy as np
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gym

from gameplr.policies import MLP
from simple_games.envs import SimEnv, TicTacToeEnv, HexapawnEnv

class DQN():
    def __init__(self, input_dim=18, output_dim=9, hid_dim=[32,32], epochs=1000):

        # input/output and model architecture parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        
        # params descrbing i/o adapters
        self.input_adapter = {}
        self.output_adapter = {}
        self.make_envs = [HexapawnEnv] #[TicTacToeEnv] #SimEnv, HexapawnEnv]

        # hyperparameters (manually set)
        self.min_eps = torch.Tensor(np.array(0.005))
        self.eps = torch.Tensor(np.array(0.95))
        self.lr = 3e-4
        self.lr_decay = 1 - 1e-2
        self.min_lr = 1e-7
        self.eps_decay = torch.Tensor(np.array(1.0 - 1e-2))
        self.batch_size = 500
        self.update_qt = 50

        self.epochs = epochs
        self.steps_per_epoch = 10000
        self.device = torch.device("cpu")
        self.discount = 0.9

        # action-value networks
        self.q = MLP(input_dim, output_dim, hid_dim=hid_dim, act=nn.LeakyReLU) \
                #Tanh)
        try:
            fpath = "q_weights_exp{}.h5".format(exp_name) 
            print("should now restoring weights from: ", fpath) 
            self.q.load_state_dict(torch.load(fpath))
            print("restoring weights from: ", fpath) 
        except:
            pass

        self.qt = MLP(input_dim, output_dim, hid_dim=hid_dim, act=nn.LeakyReLU)
        self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))

        self.q = self.q.to(self.device)
        self.qt = self.qt.to(self.device)
        for param in self.qt.parameters():
            param.requires_grad = False
        
    def change_env(self, make_env, my_seed=42):

        self.env = make_env()
        obs = self.env.reset()
        self.seed = my_seed
        torch.manual_seed = my_seed

        self.act_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]

        if self.obs_dim not in self.input_adapter.keys():
            #self.input_adapter[self.obs_dim] = torch.randn(self.obs_dim,\
            #        self.input_dim) #, requires_grad=False) 
            self.input_adapter[self.obs_dim] = torch.eye(self.obs_dim)

        if self.act_dim not in self.output_adapter.keys():
            #self.output_adapter[self.act_dim] = torch.randn(self.output_dim,\
            #        self.act_dim) #, requires_grad=False) 
            self.output_adapter[self.act_dim] = torch.eye(self.act_dim)


    def compute_q_loss(self, l_obs, l_act, l_rew, l_next_obs, l_done,\
            double=True):

        with torch.no_grad():

            #l_next_input = torch.matmul(l_next_obs, \
            #        self.input_adapter[self.obs_dim])
            #qt_out = self.qt.forward(l_next_input)
            #qt = torch.matmul(qt_out, self.output_adapter[self.act_dim])
            qt = self.qt.forward(l_next_obs)
            if double:
                #qtq_out = self.q.forward(l_next_input)
                #qtq = torch.matmul(qtq_out, self.output_adapter[self.act_dim])
                qtq = self.q.forward(l_next_obs)
                qt_max = torch.gather(qt, -1,\
                        torch.argmax(qtq, dim=-1).unsqueeze(-1))
            else:
                qt_max = torch.gather(qt, -1, \
                        torch.argmax(qt, dim=-1).unsqueeze(-1))

            yj = l_rew + ((1-l_done) * self.discount * qt_max)

        #l_input = torch.matmul(l_obs, \
        #        self.input_adapter[self.obs_dim])
        l_act = l_act.long()
        q_av_out = self.q.forward(l_obs)
        q_av = torch.matmul(q_av_out, self.output_adapter[self.act_dim])
        q_act = torch.gather(q_av, -1, l_act)

        loss =  torch.mean(torch.pow(yj - q_act, 2))

        return loss


    def get_episodes(self,steps=None):
        
        if steps == None:
            steps = self.steps_per_epoch

        l_obs = torch.Tensor()
        l_rew = torch.Tensor()
        l_act = torch.Tensor()
        l_done = torch.Tensor()
        l_next_obs = torch.Tensor()
        done = True

        with torch.no_grad():
            for step in range(steps):
                if done:
                    obs = self.env.reset()
                    obs = torch.Tensor(obs.ravel()).unsqueeze(0)
                    done = False

                
                if torch.rand(1) < self.eps:
                    action = self.env.action_space.sample()
                else:
                    # input/output adapter magic here
                    #my_input = torch.matmul(obs,self.input_adapter[\
                    #        self.obs_dim])
                    #my_output = self.q(my_input)
                    #q_values = torch.matmul(my_output, self.output_adapter[\
                    #        self.act_dim])
                    q_values = self.q(obs)

                    act = torch.argmax(q_values, dim=-1)
                    # detach action to send it to the environment
                    action = act.detach().numpy()[0]

                prev_obs = obs
                obs, reward, done, info = self.env.step(action)
                obs = torch.Tensor(obs.ravel()).unsqueeze(0)

                # Face a random policy opponent
                if not done:
                    if type(self.env.legal_moves[0] == list):
                        opponent_action = np.random.choice(\
                                self.env.legal_moves[1])
                    else:
                        opponent_action = np.random.choice(self.env.legal_moves)
                    op_obs, op_r, op_d, op_i = \
                            self.env.step(opponent_action, player=1)
                    if op_r:
                        done=True
                # concatenate data from current step to buffers
                l_obs = torch.cat([l_obs, prev_obs], dim=0)
                l_rew = torch.cat([l_rew, torch.Tensor(np.array(1.* reward))\
                        .reshape(1,1)], dim=0)
                l_act = torch.cat([l_act, torch.Tensor(np.array([action]))\
                        .reshape(1,1)], dim=0)
                l_done = torch.cat([l_done, torch.Tensor(np.array(1.0*done))\
                        .reshape(1,1)], dim=0)

                l_next_obs = torch.cat([l_next_obs, obs], dim=0)

                
        return l_obs, l_act, l_rew, l_next_obs, l_done

    def train(self, exp_name, start_epoch):
        
        # initialize optimizer
        optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',\
                factor=0.5, patience=20, verbose=True, min_lr = self.min_lr)

        self.rewards = np.array([])
        self.losses = np.array([]) 

        batch_start=0
        for epoch in range(start_epoch, start_epoch + self.epochs):
            for make_env in self.make_envs:
                self.change_env(make_env)
                # get episodes
                l_obs, l_act, l_rew, l_next_obs, l_done = self.get_episodes()
                loss_mean = 0.
                batches = 0
                for batch_start in range(0,len(l_obs)-self.batch_size,\
                        self.batch_size):

                    ll_obs = l_obs[batch_start:batch_start+self.batch_size]
                    ll_act = l_act[batch_start:batch_start+self.batch_size]
                    ll_rew = l_rew[batch_start:batch_start+self.batch_size]
                    ll_next_obs = \
                            l_next_obs[batch_start:batch_start+self.batch_size]
                    ll_done = l_done[batch_start:batch_start+self.batch_size]


                    self.q.zero_grad()
                    loss = self.compute_q_loss(ll_obs, ll_act, ll_rew, \
                            ll_next_obs, ll_done)
                    loss.backward()
                    optimizer.step()

                    batches += 1.0
                    loss_mean += loss
                temp_r = ((torch.sum(l_rew)/\
                        (torch.Tensor(np.array(1.)) + torch.sum(l_done)))\
                        .detach().cpu().numpy()).reshape(1,)
                loss_mean = loss_mean.detach().cpu().reshape(1,)

                self.rewards = np.append(self.rewards,temp_r,axis=0)
                self.losses = np.append(self.losses,loss_mean/batches,axis=0)

                scheduler.step(self.losses[-1])

                # attenuate epsilon
                self.eps = torch.max(self.min_eps, self.eps*self.eps_decay)
                print("epoch {} mean episode rewards: {}, and q loss {}".format(\
                        epoch, self.rewards[-1], self.losses[-1]))
                print("            current epsilon: {}".format(self.eps))
                        
            # maybe update qt
            if epoch % self.update_qt == 0:

                self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))
                for param in self.qt.parameters():
                    param.requires_grad = False
            if epoch % 100 == 0:
                torch.save(self.q.state_dict(),\
                        "results/q_weights_exp{}_start{}_pt2.h5"\
                        .format(exp_name, start_epoch))

                
                np.save("./results/exp{}_rewards_start{}.npy"\
                        .format(exp_name, start_epoch), np.array(self.rewards))

        fpath = "q_weights_exp{}.h5".format(exp_name)
        print("saving weights to ", fpath)
        torch.save(self.q.state_dict(),fpath)

        #results = test_policy(self.env, self.obs_dim, self.act_dim,\
        #    self.hid_dim, fpath=fpath)
        
        #np.save("./results/{}/test_{}_epoch{}.npy".format(exp_name,exp_name,\
        #        epoch),results)

if __name__ == "__main__":


    make_env = TicTacToeEnv
    exp_name = "test_exp2"
    start_epoch = 0
    
    dqn = DQN(hid_dim=[256,128,64],epochs=5000)
    dqn.change_env(make_env)
    dqn.train(exp_name, start_epoch)
    import pdb; pdb.set_trace()
