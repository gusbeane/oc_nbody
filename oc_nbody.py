import numpy as np
from amuse.units import units, constants, nbody_system
from amuse.couple.bridge import Bridge
from amuse.ic.kingmodel import new_king_model
from tqdm import tqdm
from amuse.community.ph4.interface import ph4

#import gala.potential as gp
#import gala.dynamics as gd
import astropy.units as u
from astropy.table import Table

from rbf.basis import phs3

import sys

from oceanic.oc_code import oc_code
from oceanic.options import options_reader

from oceanic.gizmo_interface import gizmo_interface
from oceanic.oceanic_io import snapshot_reader

import pickle

def evolve_cluster_in_galaxy(options_file):

    opt = options_reader(options_file)

    timestep = opt.options['timestep'] | units.Myr
    tend = opt.options['tend'] | units.Myr

    snap_reader = snapshot_reader(opt)

    cluster = oc_code(opt)
    cluster_code = cluster.code
    galaxy_code = gizmo_interface(opt)

    stars = cluster_code.particles.copy()
    starpos = galaxy_code.chosen_position_z0
    starvel = galaxy_code.chosen_velocity_z0
    
    stars.x += starpos[0] | units.kpc
    stars.y += starpos[1] | units.kpc
    stars.z += starpos[2] | units.kpc
    stars.vx = starvel[0] | units.kms
    stars.vy = starvel[1] | units.kms
    stars.vz = starvel[2] | units.kms
    
    channel = stars.new_channel_to(cluster_code.particles)
    channel.copy_attributes(["x","y","z","vx","vy","vz"])

    system = Bridge(timestep=timestep, use_threading=False)
    system.add_system(cluster_code, (galaxy_code,))
    system.add_system(galaxy_code)

    times = np.arange(0.|units.Myr, tend, timestep)
    for i,t in enumerate(tqdm(times)):
        system.evolve_model(t,timestep=timestep)
        cluster.clean_ejections(system)
        snap_reader.process_snapshot(system, galaxy_code, t)


    cluster_code.stop()

if __name__ == '__main__':
    import sys
    # options = options_reader(sys.argv[1])
    evolve_cluster_in_galaxy(sys.argv[1])
