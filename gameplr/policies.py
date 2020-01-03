import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KLineMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=[64,64], act=torch.nn.ReLU,\
            dropout_rate=0.5):
        super(KLineMLP, self).__init__()

        self.dropout_rate = dropout_rate
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim

        if type(act) is not list:

            act = [act] * len(self.hid_dim)

        self.mlp = nn.Sequential(nn.Linear(in_dim,hid_dim[0]),\
                act[0]())

        for layer in range(1,len(self.hid_dim)):
            self.mlp.add_module("layer{}".format(layer), nn.Sequential(\
                    nn.Linear(self.hid_dim[layer-1], self.hid_dim[layer]),\
                    act[layer](), nn.Dropout(p=self.dropout_rate))) 


        # add final layer
        self.mlp.add_module("output_layer", nn.Sequential(\
                nn.Linear(self.hid_dim[-1], self.out_dim)))

    def forward(self, x, my_seed=42):

        self.train, self.training = True, True
        #import pdb; pdb.set_trace()
        old_seed = torch.initial_seed()
        torch.manual_seed(my_seed)
        output = self.mlp(x)
#        torch.manual_seed(old_seed)
        
        torch.seed()

        return output
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=[64,64], act=torch.nn.ReLU):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim

        if type(act) is not list:

            act = [act] * len(self.hid_dim)

        self.mlp = nn.Sequential(nn.Linear(in_dim,hid_dim[0]),\
                act[0]())

        for layer in range(1,len(self.hid_dim)):
            self.mlp.add_module("layer{}".format(layer), nn.Sequential(\
                    nn.Linear(self.hid_dim[layer-1], self.hid_dim[layer]),\
                    act[layer]()))

        # add final layer
        self.mlp.add_module("output_layer", nn.Sequential(\
                nn.Linear(self.hid_dim[-1], self.out_dim)))

    def forward(self, x):
        return self.mlp(x)

if __name__ == "__main__":

    mlp = MLP(in_dim = 256, out_dim = 10, hid_dim = [64,32,28], act=[nn.Tanh, nn.ReLU, nn.LeakyReLU])

    x = torch.randn(1024, 256)
    y = torch.randn(1024, 10)
    steps = 10
    losses = []
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for step in range(steps):
        mlp.zero_grad()

        p = mlp(x)
        loss = torch.mean(torch.pow(y-p, 2)) 
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print("loss: ", loss)

        losses.append(loss)

    assert losses[0] > losses[-1], "expected mlp to fit random data"


    mlp = KLineMLP(32,3)
    

