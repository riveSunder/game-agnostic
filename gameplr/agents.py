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
    def __init__(self, input_dim=32, output_dim=16,\
            hid_dim=[32,32], agnostic=False, epochs=1000):


        self.hid_dim = hid_dim
        
        # params descrbing i/o adapters
        self.input_adapter = {}
        self.output_adapter = {}

        # input/output and model architecture parameters
        self.agnostic = agnostic
        if agnostic:
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.agnostic = agnostic
            self.make_envs = [TicTacToeEnv]
            self.setup_envs()
        else:
            self.make_envs = [TicTacToeEnv]
            self.setup_envs()
            self.input_dim = self.obs_dim[self.env_names[0]]
            self.output_dim = self.act_dim[self.env_names[0]]


        # hyperparameters (manually set)
        self.min_eps = torch.Tensor(np.array(0.03))
        self.eps = torch.Tensor(np.array(0.90))
        self.lr = 3e-4
        self.lr_decay = 1 - 3e-3
        self.min_lr = 1e-7
        self.eps_decay = torch.Tensor(np.array(1.0 - 1e-2))
        self.batch_size = 500
        self.update_qt = 10

        self.epochs = epochs
        self.steps_per_epoch = {'HexapawnEnv': 5000,\
                            'TicTacToeEnv': 7500,\
                            'SimEnv': 20000}
        self.device = torch.device("cpu")
        self.discount = 0.9

        # action-value networks
        self.q = [None] * 2
        self.qt = [None] * 2
        for player in [0,1]:

            self.q[player] = MLP(self.input_dim, self.output_dim, hid_dim=hid_dim, act=nn.LeakyReLU) \
                    #Tanh)
            try:
                fpath = "q_weights_exp{}.h5".format(exp_name) 
                print("should now restoring weights from: ", fpath) 
                self.q[player].load_state_dict(torch.load(fpath))
                print("restoring weights from: ", fpath) 
            except:
                pass

            self.qt[player] = MLP(self.input_dim, self.output_dim, hid_dim=hid_dim, act=nn.LeakyReLU)
            self.qt[player].load_state_dict(copy.deepcopy(self.q[player].state_dict()))

            self.q[player] = self.q[player].to(self.device)
            self.qt[player] = self.qt[player].to(self.device)
            for param in self.qt[player].parameters():
                param.requires_grad = False
        
    def change_env(self):
        pass

    def setup_envs(self, my_seed=42):

        self.envs = {}
        self.env_names = []
        self.act_dim = {}
        self.obs_dim = {}

        self.input_adapter = {}
        self.output_adapter = {}

        for make_env in self.make_envs:
            env = make_env()
            env_name = str(env).split()[0].split('.')[-1]
            self.env_names.append(env_name)
            self.envs[env_name] = env
            obs = self.envs[env_name].reset()
            self.seed = my_seed
            torch.manual_seed = my_seed

            self.act_dim[env_name] = self.envs[env_name].action_space.n
            self.obs_dim[env_name] = self.envs[env_name].observation_space.shape[0]

            if self.agnostic:
                # use random linear transformations as adapters for agnosticism
                self.input_adapter[env_name] = torch.randn(\
                        self.obs_dim[env_name],\
                        self.input_dim) 
                self.output_adapter[env_name] = torch.randn(\
                        self.output_dim,\
                        self.act_dim[env_name]) 
            else:
                # if not training for agnostic learning, use identity matrices
                self.input_adapter[env_name] = torch.eye(self.obs_dim[env_name])
                self.output_adapter[env_name] = torch.eye(self.act_dim[env_name])

    def compute_q_loss(self, l_obs, l_act, l_rew, l_next_obs, l_done,\
            player=0, double=True):

        env_name = self.env_name
        with torch.no_grad():

            l_next_input = torch.matmul(l_next_obs, \
                    self.input_adapter[env_name])
            qt_out = self.qt[player].forward(l_next_input)
            qt = torch.matmul(qt_out, self.output_adapter[env_name])
            if double:
                qtq_out = self.q[player].forward(l_next_input)
                qtq = torch.matmul(qtq_out, self.output_adapter[env_name])
                qt_max = torch.gather(qt, -1,\
                        torch.argmax(qtq, dim=-1).unsqueeze(-1))
            else:
                qt_max = torch.gather(qt, -1, \
                        torch.argmax(qt, dim=-1).unsqueeze(-1))

            yj = l_rew + ((1-l_done) * self.discount * qt_max)

        l_input = torch.matmul(l_obs, \
                self.input_adapter[env_name])
        l_act = l_act.long()
        q_av_out = self.q[player].forward(l_input)
        q_av = torch.matmul(q_av_out, self.output_adapter[env_name])
        q_act = torch.gather(q_av, -1, l_act)

        loss =  torch.mean(torch.pow(yj - q_act, 2))

        return loss


    def get_episodes(self,player=0,steps=None):
        

        l_obs = torch.Tensor()
        l_rew = torch.Tensor()
        l_act = torch.Tensor()
        l_done = torch.Tensor()
        l_next_obs = torch.Tensor()
        done = True

        env_name = self.env_name
        if steps == None:
            steps = self.steps_per_epoch[env_name]

        with torch.no_grad():
            for step in range(steps):

                if done:
                    obs = self.env.reset()
                    obs = torch.Tensor(obs.ravel()).unsqueeze(0)
                    done = False

                # Face off against opposing player
                if player:
                    if torch.rand(1) < self.eps:
                        legal_moves = self.env.legal_moves if type(self.env.legal_moves[0]) is not list else self.env.legal_moves[int(not(player))]
                        op_act = np.random.choice(legal_moves)
                    else:
                        op_input = torch.matmul(obs,\
                                self.input_adapter[self.env_name])
                        op_q_output = self.qt[not(player)](op_input)
                        op_q_values = torch.matmul(op_q_output, \
                                self.output_adapter[env_name])
                        op_act = torch.argmax(op_q_values,dim=-1)
                        op_act = op_act.detach().numpy()[0]
                    op_obs, op_r, op_d, op_i = \
                            self.env.step(op_act, player=int(not(player)))
                    if op_r:
                        done=True
                    elif op_d:
                        if env_name is not "SimEnv":
                            obs = self.env.reset()
                            obs = torch.Tensor(obs.ravel()).unsqueeze(0)
                            done = False
                        else:
                            done = True
                            l_done[-1] = torch.Tensor(np.array(1.0))
                            l_rew[-1] = torch.Tensor(np.array(1.0))



                if len(self.env.legal_moves) == 0: done = True

                if not done: 
                    if torch.rand(1) < self.eps:
                        action = self.env.action_space.sample()

                        legal_moves = self.env.legal_moves if type(self.env.legal_moves[0]) is not list else self.env.legal_moves[player]

                        action = np.random.choice(legal_moves)
                    else:
                        # input/output adapter magic here
                        my_input = torch.matmul(obs,\
                                self.input_adapter[env_name])
                        my_output = self.q[player](my_input)
                        q_values = torch.matmul(my_output, \
                                self.output_adapter[env_name])
                        act = torch.argmax(q_values, dim=-1)
                        # detach action to send it to the environment
                        action = act.detach().numpy()[0]

                    prev_obs = obs
                    obs, reward, done, info = self.env.step(action, \
                            player=player)

                    obs = torch.Tensor(obs.ravel()).unsqueeze(0)

                    # Face off against opposing player
                    if not done and not player and len(self.env.legal_moves) >0:
                        if torch.rand(1) < self.eps:
                            legal_moves = self.env.legal_moves if type(self.env.legal_moves[0]) is not list else self.env.legal_moves[int(not(player))]
                            op_act = np.random.choice(legal_moves)
                        else:
                            op_input = torch.matmul(obs,\
                                    self.input_adapter[self.env_name])
                            op_q_output = self.qt[not(player)](op_input)
                            op_q_values = torch.matmul(op_q_output, \
                                    self.output_adapter[env_name])
                            op_act = torch.argmax(op_q_values,dim=-1)
                            op_act = op_act.detach().numpy()[0]
                        op_obs, op_r, op_d, op_i = \
                                self.env.step(op_act, player=int(not(player)))
                        if op_r:
                            done=True
                        elif op_d:
                            if env_name is not "SimEnv":
                                done = True
                            else:
                                done = True
                                reward = 1.0

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
        optimizer = [None] * 2
        scheduler = [None] * 2
        for player in [0,1]:
            optimizer[player] = torch.optim.Adam(\
                    self.q[player].parameters(), lr=self.lr)
            scheduler[player] = ReduceLROnPlateau(optimizer[player], mode='min',\
                factor=0.5, patience=50, verbose=True, min_lr = self.min_lr)

        self.rewards = np.array([])
        self.losses = np.array([]) 

        batch_start=0
        for epoch in range(start_epoch, start_epoch + self.epochs):
            for env_name in self.env_names: 
                self.env_name = env_name
                self.env = self.envs[env_name]

                for player in [0,1]:
                    # get episodes
                    l_obs, l_act, l_rew, l_next_obs, l_done = \
                            self.get_episodes(player=player)
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


                        self.q[player].zero_grad()
                        loss = self.compute_q_loss(ll_obs, ll_act, ll_rew, \
                                ll_next_obs, ll_done, player=player)
                        loss.backward()
                        optimizer[player].step()

                        batches += 1.0
                        loss_mean += loss
                    temp_r = ((torch.sum(l_rew)/\
                            (torch.Tensor(np.array(1.)) + torch.sum(l_done)))\
                            .detach().cpu().numpy()).reshape(1,)
                    loss_mean = loss_mean.detach().cpu().reshape(1,)

                    self.rewards = np.append(self.rewards,temp_r,axis=0)
                    self.losses = np.append(self.losses,loss_mean/batches,axis=0)

                    scheduler[player].step(self.losses[-1])

                    # attenuate epsilon
                    print("{} epch {} plyr {} wins: {:.3f}/{:.3f}".format(\
                            env_name, epoch, player, np.sum(l_rew.numpy()), \
                            np.sum(l_done.numpy())))
                    print("q loss: {:.3e}     current epsilon: {:.3f}"\
                            .format(self.losses[-1], self.eps))
                            
            self.eps = torch.max(self.min_eps, self.eps*self.eps_decay)
            # maybe update qt
            if epoch % self.update_qt == 0:
                print("updating q_t . . .")

                self.q[player].load_state_dict(copy.deepcopy(self.q[player].state_dict()))
                for param in self.qt[player].parameters():
                    param.requires_grad = False
            if epoch % 100 == 0:
                torch.save(self.q[player].state_dict(),\
                        "results/q_weights_exp{}_start{}_pt2.h5"\
                        .format(exp_name, start_epoch))

                
                np.save("./results/exp{}_qlosses_start{}.npy"\
                        .format(exp_name, start_epoch), np.array(self.losses))
                np.save("./results/exp{}_rewards_start{}.npy"\
                        .format(exp_name, start_epoch), np.array(self.rewards))

        fpath = "q_weights_exp{}.h5".format(exp_name)
        print("saving weights to ", fpath)
        torch.save(self.q[player].state_dict(),fpath)
        np.save("./results/exp{}_qlosses_start{}.npy"\
                .format(exp_name, start_epoch), np.array(self.losses))
        np.save("./results/exp{}_rewards_start{}.npy"\
                .format(exp_name, start_epoch), np.array(self.rewards))

        #results = test_policy(self.env, self.obs_dim, self.act_dim,\
        #    self.hid_dim, fpath=fpath)
        
        #np.save("./results/{}/test_{}_epoch{}.npy".format(exp_name,exp_name,\
        #        epoch),results)

if __name__ == "__main__":


    make_env = TicTacToeEnv

    for exp_name, agnostic in zip(["agnostic_exp","id_exp"],[True,False]):
        start_epoch = 0
        dqn = DQN(hid_dim=[256,128,64],epochs=5000, agnostic=agnostic)

        dqn.train(exp_name, start_epoch)
    import pdb; pdb.set_trace()
    
