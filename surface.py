import numpy as np
import torch as t

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from model import *
from copy import deepcopy
from time import time

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
        x_bound=(-6,6),
        y_bound=(-6,6),
        resolution=250,
    ):
    with t.no_grad():
        x_values = np.linspace(*x_bound,resolution)
        y_values = np.linspace(*y_bound,resolution)
        surface = np.empty((len(x_values),len(y_values)))

        w = net.get_nth_layer(layer,sub_layer).data

        for i,x in enumerate(x_values):
            for j,y in enumerate(y_values):
                w[0] = x
                w[1] = y
                outputs = net.forward(inputs).detach()
                surface[i,j] = ((outputs - targets)**2).mean().item()
                #surface[i,j] = t.nn.MSELoss()(outputs,targets).item()

    return x_values,y_values,surface # surface is really just z_values

def descend_gradient(
        net,
        inputs,
        targets,
        layer,
        sub_layer=0,
        lr=0.002,
        momentum=0.9,
        epochs=8000,
        size=10,
        orbit_speed=0.05
    ):
    try:
        scatter = gl.GLScatterPlotItem(size=size,color=(1,0,0,1))#red
        path = gl.GLLinePlotItem(color=(1,1,1,.4))
        view.addItem(scatter)
        view.addItem(path)

        points = np.empty((epochs,3))
        colors = np.zeros((epochs,4))
        colors[:,3] = 1 # opacity

        w = net.get_nth_layer(layer,sub_layer)
        w.data[:] = t.rand(w.shape) *8 -4
        # lock all weight except visible layer
        for i,param in enumerate(net.parameters()):
            if i != layer:
                param.requires_grad = False

        for i in range(epochs):
            loss = net._one_epoch(inputs,targets,lr=lr,momentum=momentum)

            points[i,:2] = w.detach().numpy()
            #points[i,2] = loss.item()
            outputs = net.forward(inputs).detach()
            points[i,2] = ((outputs - targets)**2).mean().item()

            colors[:i+1,1] = np.linspace(0,1,i+1)
            colors[:i+1,0] = np.linspace(1,0,i+1)
            scatter.setData(pos=points[:i+1], color=colors[:i+1])
            path.setData(pos=points[:i+1])

            pg.QtGui.QApplication.processEvents()
            view.orbit(orbit_speed,0)

            if view.isHidden(): return # game over

    except KeyboardInterrupt: # also game over
        print('Training halted')

def run_simulation(
        nn,
        layer,
        sub_layer=0,
        n = 100,
        resolution = 200,
        open_window = "max",
        orbit_speed = .05,
        ):
    if open_window == "full":
        view.showFullScreen()
    elif open_window == "max":
        view.showMaximized()
    elif open_window:
        view.show()

    data = gen_data(n)

    x,y,z = create_loss_surface(
                nn,
                *data,
                layer,
                sub_layer,
                resolution=resolution,
            )
    surface.setData(x=x, y=y, z=z)

    stop = time() + 3
    while not view.isHidden() and time() < stop:
        pg.QtGui.QApplication.processEvents()

    while not view.isHidden():
        descend_gradient(nn, *data, layer, sub_layer, epochs=80,
                orbit_speed=orbit_speed)
        view.orbit(orbit_speed,0)
        pg.QtGui.QApplication.processEvents()

def main():
    nn = t.load('xsquared_network0.93')
    run_simulation(
            nn,
            layer=0,
            sub_layer=1
    )

if __name__ == "__main__": main()
