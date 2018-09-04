import numpy as np
from amuse.units import units, constants, nbody_system
from amuse.ext.bridge import bridge
from amuse.community.ph4.interface import ph4
from amuse.community.bhtree.interface import BHTree
from amuse.ic.kingmodel import new_king_model
from oceanic.options import options_reader

class oc_code(object):
    #def __init__(self, nbodycode, N, W0, Mcluster, Rcluster, ncore, parameters = []):
    def __init__(self, options_reader):

        options_reader.set_options(self)
        # self.N, self.W0, self.Mcluster, self.Rcluster, self.ncpu
        # self.gpu_enabled, self.ngpu, self.softening

        self.converter = nbody_system.nbody_to_si(self.Mcluster | units.MSun,
                                                 self.Rcluster | units.parsec)
        self.bodies = new_king_model(self.N, self.W0, convert_nbody=self.converter)

        if self.gpu_enabled:
            self.code = self.nbodycode(self.converter, number_of_workers=self.ngpu, mode='gpu')
        else:
            self.code = self.nbodycode(self.converter, number_of_workers=self.ncpu)

        parameters = {'epsilon_squared': (self.softening | units.parsec)**2}

        for name, value in parameters.items():
            setattr(self.code.parameters, name, value)
        self.code.particles.add_particles(self.bodies)

    def clean_ejections(self, system):
        cluster_positions = np.transpose([system.particles.x.value_in(units.parsec), 
            system.particles.y.value_in(units.parsec), system.particles.z.value_in(units.parsec)])
        median_position = np.median(cluster_positions, axis=0)
        dist = np.linalg.norm(np.subtract(cluster_positions, median_position), axis=1)
        keys = np.where(dist > self.eject_cut)[0]
        if len(keys>1):
            system.particles.remove_particles(system.particles[keys])
        elif len(keys==1):
            system.particles.remove_particle(system.particles[keys])
        else:
            return None

if __name__ == '__main__':
    import sys
    from amuse.lab import Particle
    options_file = sys.argv[1]
    opt = options_reader(options_file)
    oc = oc_code(opt)

    p = Particle()
    p.x = 1 | units.kpc
    p.y = 0 | units.kpc
    p.z = 0 | units.kpc

    p.vx = 0 | units.kms
    p.vy = 0 | units.kms
    p.vz = 0 | units.kms

    p.mass = 1 | units.MSun

    oc.code.particles.add_particle(p)