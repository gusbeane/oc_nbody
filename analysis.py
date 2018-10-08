import matplotlib; matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import agama
import gizmo_analysis as gizmo
import numpy as np

import sys



class agama_wrapper(object):
    def __init__(self, opt):
        opt.set_options(self)
        agama.setUnits(mass=1, length=1, velocity=1)

    def update_index(self, index, ss_id=None, snap=None):
        agama.setUnits(mass=1, length=1, velocity=1)
        self.current_index = index
        self.ss_id = ss_id
        if snap is not None:
            self.snap = snap
        else:
            self.snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                                     'index', index,
                                                     properties=['id',
                                                                 'position',
                                                                 'velocity',
                                                                 'mass',
                                                                 'form.scale⁠factor'],
                                            simulation_directory=
                                            self.simulation_directory,
                                            assign_principal_axes=True)
        star_position = self.snap['star'].prop('host.distance.principal')
        gas_position = self.snap['gas'].prop('host.distance.principal')
        dark_position = self.snap['dark'].prop('host.distance.principal')

        star_mass = self.snap['star']['mass']
        gas_mass = self.snap['gas']['mass']
        dark_mass = self.snap['dark']['mass']



        position = np.concatenate((star_position, gas_position))
        mass = np.concatenate((star_mass, gas_mass))

        self.pdark = agama.Potential(type="Multipole",
                                     particles=(dark_position, dark_mass),
                                     symmetry='a', gridsizeR=20, lmax=2)
        self.pbar = agama.Potential(type="CylSpline",
                                    particles=(position, mass),
                                    gridsizer=20, gridsizez=20,
                                    mmax=0, Rmin=0.2,
                                    Rmax=50, Zmin=0.02, Zmax=10)
        self.potential = agama.Potential(self.pdark, self.pbar)

        if ss_id is not None:
            self.ss_init = True
            ss_key = np.where(self.snap['star']['id'] == self.ss_id)[0]
            self.chosen_position = self.snap['star'].prop(
                'host.distance.principal')[ss_key]
            self.chosen_velocity = self.snap['star'].prop(
                'host.velocity.principal')[ss_key]
        else:
            self.ss_init = False

        self.af = agama.ActionFinder(self.potential, interp=False)

    def update_ss(self, ss_id):
        self.ss_init = True
        ss_key = np.where(self.snap['star']['id'] == ss_id)[0]
        self.chosen_position = self.snap['star'].prop(
            'host.distance.principal')[ss_key]
        self.chosen_velocity = self.snap['star'].prop(
            'host.velocity.principal')[ss_key]

    def actions(self, poslist, vlist, add_ss=False, in_kpc=False):
        if not in_kpc:
            poslist /= 1000.0
        if add_ss:
            if not self.ss_init:
                raise Exception('need to initialize with a ss id to add \
                                 ss pos, vel')
            else:
                poslist = np.add(poslist, self.chosen_position)
                vlist = np.add(vlist, self.chosen_velocity)
        points = np.c_[poslist, vlist]
        return self.af(points) # Jr, Jz, Lz

    def ss_action(self):
        points = np.c_[self.chosen_position, self.chosen_velocity]
        return self.af(points)[0]

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
        this_mass = self.snapshots[frame]['mass']
        # data = np.array([this_x_data, this_y_data])
        scat.set_offsets(np.c_[this_x_data, this_y_data])
        scat.set_sizes(this_mass)
        t = self.snapshots[frame]['time']
        self.ax.set_title("{:.2f}".format(t))
        return (scat,)

    def __call__(self):
        self.animation = FuncAnimation(self.fig, self._animate_,
                                       np.arange(self.start, self.end),
                                       fargs=[self.scat],
                                       interval=1000.0/self.fps,
                                       blit=False)
        self.animation.save(self.fileout, dpi=600)


if __name__ == '__main__':
    from oceanic.options import options_reader
    opt = options_reader(sys.argv[1])
    ag = agama_wrapper(opt)
