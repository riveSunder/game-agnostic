
import numpy as np
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F

import gym

from gameplr.policies import MLP
from simple_games.envs import SimEnv, TicTacToeEnv, HexapawnEnv

class DQN():
    def __init__(self, input_dim, output_dim, hid_dim=[32,32], epochs=1000):

        # input/output and model architecture parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        
        # params descrbing i/o adapters
        self.input_adapter = {}
        self.output_adapter = {}

        # hyperparameters (manually set)
        self.min_eps = 0.01
        self.eps = 0.95
        self.eps_decay = 1e-2
        self.lr = 3e-4
        self.batch_size = 512
        self.update_qt = 200

        self.epochs = epochs
        self.device = torch.device("cpu")
        self.discount = 0.9

        # action-value networks
        self.q = MLP(obs_dim, act_dim, hid_dim=hid_dim, act=nn.Tanh)
        try:
            fpath = "q_weights_exp{}.h5".format(exp_name) 
            print("should now restoring weights from: ", fpath) 
            self.q.load_state_dict(torch.load(fpath))
            print("restoring weights from: ", fpath) 
        except:
            pass

        self.qt = MLP(obs_dim, act_dim, hid_dim=hid_dim, act=nn.Tanh) 
        self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))

        self.q = self.q.to(self.device)
        self.qt = self.qt.to(self.device)
        for param in self.qt.parameters():
            param.requires_grad = False
        
    def change_env(self, env, my_seed=42):

        self.env = env
        self.seed = seed
        torch.manual_seed = my_seed

        self.act_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]

        if self.obs_dim not in self.input_adapter.keys():
            self.input_adapter[self.obs_dim] = torch.randn(self.obs_dim,\
                    self.input_dim) #, requires_grad=False) 

        if self.act_dim not in self.output_adapter.keys():
            self.input_adapter[self.act_dim] = torch.randn(self.output_dim,\
                    self.act_dim) #, requires_grad=False) 


    def compute_q_loss(self, l_obs, l_act, l_rew, l_next_obs, l_done,\
            double=True):

        with torch.no_grad():
            qt = self.qt.forward(l_next_obs)
            if double:
                qtq = self.q.forward(l_next_obs)
                qt_max = torch.gather(qt, -1,\
                        torch.argmax(qtq, dim=-1).unsqueeze(-1))
            else:
                qt_max = torch.gather(qt, -1, \
                        torch.argmax(qt, dim=-1).unsqueeze(-1))

            yj = l_rew + ((1-l_done) * self.discount * qt_max)

        l_act = l_act.long()
        q_av = self.q.forward(l_obs)
        q_act = torch.gather(q_av, -1, l_act)

        loss =  torch.mean(torch.pow(yj-q_act,2))

        return loss


if __name__ == "__main__":
    pass
