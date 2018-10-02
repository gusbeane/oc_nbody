import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

import pickle
import sys, os


class cluster_animator(object):
    def __init__(self, snapshots, xaxis='x', yaxis='y',
                 xmin=-10, xmax=10, ymin=-10, ymax=10,
                 start=None, end=None, fps=30, fileout=None):

        self.snapshots = snapshots

        self.xaxis = xaxis
        self.yaxis = yaxis

        self._xaxis_key_ = self._axis_key_(self.xaxis)
        self._yaxis_key_ = self._axis_key_(self.yaxis)

        if start is None:
            self.start = 0
        else:
            self.start = start

        if end is None:
            self.end = len(self.snapshots)
        else:
            self.end = end

        self.fps = fps

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        first_x = self.snapshots[self.start]['position'][:, self._xaxis_key_]
        first_y = self.snapshots[self.start]['position'][:, self._yaxis_key_]
        first_mass = self.snapshots[self.start]['mass']

        self.fig, self.ax = plt.subplots(1)
        self.scat = self.ax.scatter(first_x, first_y, s=first_mass)

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        self.ax.set_xlabel(self.xaxis+' [pc]')
        self.ax.set_ylabel(self.yaxis+' [pc]')
        self.fig.tight_layout()

        if fileout is None:
            self.fileout = 'movie_' + self.xaxis + '_' + str(self.xmin)
            self.fileout += '_' + str(self.xmax) + '_' + self.yaxis + '_'
            self.fileout += str(self.ymin) + '_' + str(self.ymax) + '.mp4'
        else:
            self.fileout = fileout

    def _axis_key_(self, axis):
        if axis == 'x':
            return 0
        elif axis == 'y':
            return 1
        elif axis == 'z':
            return 2
        else:
            raise Exception('cant recognize axis: '+axis)
            sys.exit(1)

    def _animate_(self, frame, scat):
        this_x_data = self.snapshots[frame]['position'][:, self._xaxis_key_]
        this_y_data = self.snapshots[frame]['position'][:, self._yaxis_key_]
        # data = np.array([this_x_data, this_y_data])
        scat.set_offsets(np.c_[this_x_data, this_y_data])
        return (scat,)

    def __call__(self):
        self.animation = FuncAnimation(self.fig, self._animate_,
                                       np.arange(self.start, self.end),
                                       fargs=[self.scat],
                                       interval=1000.0/self.fps)
        self.animation.save(self.fileout, dpi=600)


'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read in processors,
        data file.')
    parser.add_argument('-i', help='snapshot file, .npy', required=True)


    snapshot_file = sys.argv[1]
'''
