import agama
import gizmo_analysis as gizmo
import numpy as np


class agama_wrapper(object):
    def __init__(self, opt):
        opt.set_options(self)

    def update_index(self, index, chosen_id=None):
        self.current_index = index
        self.chosen_id = chosen_id
        snap = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                            'index', index,
                                            properties=['id', 'position',
                                                        'velocity', 'mass',
                                                        'form.scalefactor'],
                                            simulation_directory=
                                            self.simulation_directory,
                                            assign_principal_axes=True)
        star_position = snap['star'].prop('host.distance.principal')
        gas_position = snap['gas'].prop('host.distance.principal')
        dark_position = snap['dark'].prop('host.distance.principal')

        star_mass = snap['star']['mass']
        gas_mass = snap['gas']['mass']
        dark_mass = snap['dark']['mass']

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

        if chosen_id is not None:
            self.ss_init = True
            chosen_key = np.where(snap['star']['id'] == self.chosen_id)[0]
            self.chosen_position = snap['star'].prop(
                'host.distance.principal')[chosen_key]
            self.chosen_velocity = snap['star'].prop(
                'host.velocity.principal')[chosen_key]
        else:
            self.ss_init = False

        self.af = agama.ActionFinder(self.potential, interp=False)

    def actions(self, poslist, vlist, add_ss=False):
        if add_ss:
            if ~self.ss_init:
                raise Exception('need to initialize with a ss id to add \
                                 ss pos, vel')
            else:
                poslist = np.add(poslist, self.chosen_position)
                vlist = np.add(vlist, self.chosen_velocity)
        points = np.c_[poslist, vlist]
        return self.af(points)
