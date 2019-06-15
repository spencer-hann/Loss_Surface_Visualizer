# Loss_Surface_Visualizer

Using PyQtGraph to visualize partial loss surfaces for a simple 2-layer neural network.

## Usage:
To run the main program, simply run `python surface.py <filename>`. To create new networks to visualize, run `python model.py`. New models will be stored with a testing accuracy appended to the filename. For more detailed instructions, like how to change the learning rate, or which weights vector the loss surface is being visualized with respect to, run `python surface.py --help`. To finish running the program, simply close the window that pyqtgraph opened and the program will exit. Interacting with 3d pyqtgraph plots with a trackpad is possible, but not a smooth experience. A normal mouse is definitely recommended.
