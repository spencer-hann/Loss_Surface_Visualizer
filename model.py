import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from sklearn.metrics import accuracy_score
from itertools import islice

def gen_data(n):
    x = torch.rand(n,2) *2 -1
    y = torch.empty(n)
    for i in range(n):
        if x[i,0]**2 > x[i,1]:
            y[i] = -1
        else:
            y[i] = 1
    return x,y

class Network(nn.Module):
    _b = 1
    b = bool(_b)
    def __init__(self, bias1=b, bias2=b):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2,2, bias1)
        self.fc2 = nn.Linear(2,1, bias2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _one_epoch(self,X,Y,lr=0.001,momentum=.9):
        optimizer = optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
        )
        criterion = nn.MSELoss()
        for x,y in zip(X,Y):
            optimizer.zero_grad()

            y_hat = self(x)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        return loss

    def train(self, X, Y, epochs=801, stopping_criterion=1e-2):
        try:
            for i in range(epochs):
                loss = self._one_epoch(X,Y)

                if i % 100 == 0: print(i, loss.item())

        except KeyboardInterrupt:
            print("\nTraining halted.\n",flush=True)
            return

    def get_nth_layer(self,n,sub_layer=0):
        name,w = next(islice(self.named_parameters(),n,None))
        # 'bias' parameter is 1-dimensional
        # only equipped here to handle 'bias' and 'weight' named parameters
        if 'bias' not in name:
            w = w[sub_layer]
        return w


def p_check(n):
    print(n.parameters)
    for i,p in enumerate(n.parameters()):
        print(p.data.type())
        print(p)

def plot_data(x, y):
    plt.axvline(0,color='k',alpha=.6)
    plt.axhline(0,color='k',alpha=.6)

    for point,label in zip(x,y):
        if label > 0:
            plt.scatter(point[0],point[1],color='b')
        else:
            plt.scatter(point[0],point[1],color='r')
    plt.grid()
    plt.show()

def main(plot=False, display=True):
    n = Network()
    #p_check(n)

    x,y = gen_data(2000)
    n.train(x, y)

    y_hat = n(x).sign()
    if display:
        print(accuracy_score(y.detach(),y_hat.detach()))

    if plot:
        plot_data(x, y)

    x,y = gen_data(1000)
    y_hat = n(x).sign()
    acc = accuracy_score(y.detach(),y_hat.detach())
    if display:
        print(acc)
    if plot:
        plot_data(x,y_hat)

    torch.save(n, f"xsquared_network{acc}")

if __name__ == "__main__": main()
