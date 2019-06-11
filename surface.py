import numpy as np
from numpy.random import randint

import pyqtgraph as pg
import pyqtgraph.opengl as gl

## build a QApplication before building other widgets
pg.mkQApp()

## make a widget for displaying 3D objects
view = gl.GLViewWidget()
#view.showFullScreen()

## create three grids, add each to the view
grid = gl.GLGridItem()
view.addItem(grid)

#create surface
surface = gl.GLSurfacePlotItem()
view.addItem(surface)

def run_simulation(
        life,
        delay = .4,
        breed = False,
        k = 3,
        min_life = 3,
        max_life = 32,
        gen_allowance = 4,
        open_window = "max",
        orbit_speed = .1,
        ):
    color = (1,1,1,1) # all on / white
    size = np.zeros(life.shape)
    time_last = time()
    gen_counter = 0

    if not open_window: delay = 0

    if open_window == "full": view.showFullScreen()
    elif open_window == "max": view.showMaximized()
    elif open_window:         view.show()

    while not view.isHidden():
        view.orbit(orbit_speed,0)
        pg.QtGui.QApplication.processEvents()

def main():
    run_simulation(
                life,
                k=4,
                breed=True,
                delay=.0,
                gen_allowance=30,
                min_life=10,
                max_life=512,
                )

if __name__ == "__main__":
    main()
