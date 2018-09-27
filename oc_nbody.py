import numpy as np
from amuse.units import units
from amuse.couple.bridge import Bridge
from tqdm import tqdm

import sys

from oceanic.oc_code import oc_code
from oceanic.options import options_reader

from oceanic.gizmo_interface import gizmo_interface
from oceanic.oceanic_io import snapshot_reader


def evolve_cluster_in_galaxy(options_file):
    opt = options_reader(options_file)

    timestep = opt.options['timestep']  # in Myr
    tend = opt.options['tend']  # in Myr
    times = np.arange(0.0, tend, timestep) | units.Myr

    snap_reader = snapshot_reader(opt)

    cluster = oc_code(opt)
    cluster_code = cluster.code
    galaxy_code = gizmo_interface(opt)

    stars = cluster_code.particles.copy()

    channel = stars.new_channel_to(cluster_code.particles)
    channel.copy_attributes(["x", "y", "z", "vx", "vy", "vz"])

    system = Bridge(timestep=timestep, use_threading=False)
    system.add_system(cluster_code, (galaxy_code,))
    system.add_system(galaxy_code)

    for i, t in enumerate(tqdm(times)):
        system.evolve_model(t, timestep=timestep | units.Myr)
        cluster.clean_ejections(system)
        snap_reader.process_snapshot(system, galaxy_code, t)

    cluster_code.stop()


if __name__ == '__main__':
    evolve_cluster_in_galaxy(sys.argv[1])
