import matplotlib; matplotlib.use('agg')

import agama
import gizmo_analysis as gizmo
import numpy as np


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
        return self.af(points)


if __name__ == '__main__':
    from oceanic.options import options_reader
    import sys
    opt = options_reader(sys.argv[1])
    ag = agama_wrapper(opt)
