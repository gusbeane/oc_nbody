from amuse.units import units
import numpy as np
import dill

class meta_array(np.ndarray):
    """Array with metadata."""

    def __new__(cls, array, dtype=None, order=None, **kwargs):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        obj.meta = kwargs
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.meta = getattr(obj, 'metadata', None)

class snapshot_reader(object):
    def __init__(self, options_reader, galaxy_code):
        options_reader.set_options(self)
        self.frames = meta_array([])
        self.frames.meta['ss_id'] = galaxy_code.chosen_id
        self.frames.meta['simulation_directory'] = galaxy_code.simulation_directory

    def process_snapshot(self, system, galaxy_code, i, time, final=None):
        self.frames.append(self._grab_frame_(system, galaxy_code, time))
        if np.mod(i, self.write_frequency) == 0:
            fout = self.output_directory+'/cluster_snapshots.p'
            dill.dump(self.frames, open(fout, 'wb'))

    def finish_sim(self):
        fout = self.output_directory+'/cluster_snapshots.p'
        dill.dump(self.frames, open(fout, 'wb'))

    def _grab_frame_(self, system, galaxy_code, time):
        x = system.particles.x.value_in(units.parsec)
        y = system.particles.y.value_in(units.parsec)
        z = system.particles.z.value_in(units.parsec)

        vx = system.particles.vx.value_in(units.kms)
        vy = system.particles.vy.value_in(units.kms)
        vz = system.particles.vz.value_in(units.kms)

        mass = system.particles.mass.value_in(units.MSun)

        position = np.transpose([x, y, z])
        velocity = np.transpose([vx, vy, vz])

        chosen_position = galaxy_code.chosen_evolved_position * 1000.0
        # chosen_velocity = galaxy_code.chosen_evolved_velocity

        time = time.value_in(units.Myr)

        frame = {'time': time,
                 'position': position,
                 'velocity': velocity,
                 'chosen_position': chosen_position,
                 'mass': mass}
        # 'chosen_velocity': chosen_velocity}

        return frame
