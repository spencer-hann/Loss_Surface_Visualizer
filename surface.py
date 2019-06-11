import numpy as np
import torch as t

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from model import *
from itertools import islice

nn = t.load('xsquared_network')

## build a QApplication before building other widgets
pg.mkQApp()

## make a widget for displaying 3D objects
view = gl.GLViewWidget()
view.showMaximized()

## create flat grid and directional lines
grid = gl.GLGridItem()
view.addItem(grid)
x_dir = gl.GLLinePlotItem(
        pos=np.array([[0,0,0],[1,0,0]]),
        color=np.array([[1,0,0,1],[1,0,0,0]]),
        width=2,
)
view.addItem(x_dir)
y_dir = gl.GLLinePlotItem(
        pos=np.array([[0,0,0],[0,1,0]]),
        color=np.array([[0,1,0,1],[0,1,0,0]]),
        width=2,
)
view.addItem(y_dir)
z_dir = gl.GLLinePlotItem(
        pos=np.array([[0,0,0],[0,0,1]]),
        color=np.array([[0,0,1,1],[0,0,1,0]]),
        width=2,
)
view.addItem(z_dir)

#create surface
surface = gl.GLSurfacePlotItem(
        drawEdges=True,
        drawFaces=False,
        #computeNormals=False,
)
view.addItem(surface)

def create_loss_surface(
        net,
        inputs,
        targets,
        layer,
        sub_layer=0,
        x_bound=(-5,5),
        y_bound=(-5,5),
        resolution=250,
    ):
    x_values = np.linspace(*x_bound,resolution)
    y_values = np.linspace(*y_bound,resolution)
    surface = np.empty((len(x_values),len(y_values)))

    name,w = next(islice(net.named_parameters(),layer,None))
    # 'bias' parameter is 1-dimensional
    # only equipped here to handle 'bias' and 'weight' named parameters
    if 'bias' not in name:
        w = w[sub_layer]
    w = w.data

    for i,x in enumerate(x_values):
        for j,y in enumerate(y_values):
            w[0] = x
            w[1] = y
            outputs = net.forward(inputs).detach()
            surface[i,j] = ((outputs - targets)**2).mean().item()

    return x_values,y_values,surface # surface is really just z_values

#x = np.arange(n) - 5
#y = np.arange(n) - 5
#x = np.linspace(0,1,n)
#y = np.linspace(0,1,n)

#z = np.random.rand(n,n)
#z[:] = x
#z = z.T
#z[:] += y*shift_factor
#z = squared(x,y)
#z = simple(z)
n = 100
colors = np.random.rand(n,n,4)
#colors[:,:,3] = 1

x,y,z = create_loss_surface(
    nn,
    *gen_xor(10),
    2,
)
surface.setData(x=x, y=y, z=z, colors=colors)


def run_simulation(
        open_window = "max",
        orbit_speed = .1,
        ):
    if open_window == "full":
        view.showFullScreen()
    elif open_window == "max":
        view.showMaximized()
    elif open_window:
        view.show()

    while not view.isHidden():
        view.orbit(orbit_speed,0)
        pg.QtGui.QApplication.processEvents()

def main():
    run_simulation(orbit_speed=.0)

if __name__ == "__main__": main()
