import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=[64,64], act=torch.relu):
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
    steps = 1000
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
