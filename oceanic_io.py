from amuse.units import units
import numpy as np

class snapshot_reader(object):
    def __init__(self, options_reader):
        options_reader.set_options(self)
        self.frames = []

    def process_snapshot(self, system, galaxy_code, time):
        self.frames.append(self._grab_frame_(system, galaxy_code, time))
        np.save(self.output_directory+'/cluster_snapshots', self.frames)

    def _grab_frame_(self, system, galaxy_code, time):
        x = system.particles.x.value_in(units.parsec)
        y = system.particles.y.value_in(units.parsec)
        z = system.particles.z.value_in(units.parsec)

        vx = system.particles.vx.value_in(units.kms)
        vy = system.particles.vy.value_in(units.kms)
        vz = system.particles.vz.value_in(units.kms)

        position = np.transpose([x, y, z])
        velocity = np.transpose([vx, vy, vz])

        chosen_position = galaxy_code.chosen_evolved_position * 1000.0
        chosen_velocity = galaxy_code.chosen_evolved_velocity

        time = time.value_in(units.Myr)

        frame = {'time'    : time,
                 'position': position,
                 'velocity': velocity,
                 'chosen_position': chosen_position,
                 'chosen_velocity': chosen_velocity}

        return frame

"""
def read_first_snapshot(index, simulation_directory, ):

    head = gizmo.io.Read.read_header(snapshot_value=index, simulation_directory=simulation_directory)
    sim_name = head['simulation.name'].replace(" ","_")
    cache_name = 'first_snapshot_'+sim_name+'_index'+str(self.startnum)+'.p'
    cache_file = self.cache_directory + '/' + cache_name
    try:
        self.first_snapshot = pickle.load(open(cache_file, 'rb'))
        print('found and loaded cached file for first_snapshot:')
        print(cache_name)
    except:
        print('couldnt find cached file for first_snapshot:')
        print(cache_name)
        print('constructing...')
        self.first_snapshot = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.startnum, 
                                        simulation_directory=self.simulation_directory, assign_center=False)#,
                                        #particle_subsample_factor=20)
        

        pickle.dump(self.first_snapshot, open(cache_file, 'wb'), protocol=4)
"""