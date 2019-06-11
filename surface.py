import numpy as np
import torch as t

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from model import *

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

    w = net.get_nth_layer(layer,sub_layer).data

    for i,x in enumerate(x_values):
        for j,y in enumerate(y_values):
            w[0] = x
            w[1] = y
            outputs = net.forward(inputs).detach()
            #surface[i,j] = ((outputs - targets)**2).mean().item()
            surface[i,j] = t.nn.MSELoss()(outputs,targets).item()

    return x_values,y_values,surface # surface is really just z_values

def descend_gradient(
        net,
        inputs,
        targets,
        layer,
        lr=0.001,
        momentum=0.9,
        epochs=4000,
    ):
    try:
        scatter = gl.GLScatterPlotItem(color=(1,0,0,1))#red
        view.addItem(scatter)
        points = np.empty((epochs,3))

        # lock all weight exept visible layer
        for i,param in enumerate(net.parameters()):
            if i != layer:
                param.requires_grad = False

        w = net.get_nth_layer(layer).data
        w = t.rand(w.shape) *4 -2

        for i in range(epochs):
            loss = net._one_epoch(inputs,targets,lr=lr,momentum=momentum)

            points[i,:2] = w.detach().numpy()[:]
            points[i,2] = loss.item()
            #outputs = net.forward(inputs).detach()
            #points[i,2] = ((outputs - targets)**2).mean().item()

            scatter.setData(pos=points[:i+1,:])

            pg.QtGui.QApplication.processEvents()

            if view.isHidden(): return # game over
    except KeyboardInterrupt: # also game over
        print('Training halted')

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

def run_simulation(
        nn,
        layer,
        n = 20,
        open_window = "max",
        orbit_speed = .1,
        ):
    if open_window == "full":
        view.showFullScreen()
    elif open_window == "max":
        view.showMaximized()
    elif open_window:
        view.show()

    data = gen_data(n)

    x,y,z = create_loss_surface(nn, *data, layer,)
    surface.setData(x=x, y=y, z=z, colors=colors)

    descend_gradient(nn, *data, layer)
    while not view.isHidden():
        view.orbit(orbit_speed,0)
        pg.QtGui.QApplication.processEvents()

def main():
    nn = t.load('xsquared_network')
    run_simulation(nn,layer=0,orbit_speed=.04)

if __name__ == "__main__": main()
