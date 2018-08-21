import numpy as np
import gizmo_analysis as gizmo
import utilities as ut
from scipy import interpolate
from scipy.spatial import cKDTree
from rbf.interpolate import RBFInterpolant
from rbf.basis import phs3
from grid import grid
from amuse.units import units

class gizmo_interface(object):
    def __init__(self, directory, startnum, endnum,
                    num_prior=3):
        
        self.convert_kms_Myr_to_kpc = 20000.0*np.pi / (61478577.0) # thanks wolfram alpha

        # TODO make these user-definable
        self.nclose = 100
        self.basis = phs3
        self.order = 5
        self.rcut_max = 14
        self.rcut_min = 4

        self.directory = directory
        self.startnum = startnum
        self.endnum = endnum

        Rmin = 7.
        Rmax = 9.
        zmin = -1.
        zmax = 1.

        self.grid_rmax = (250.0 + 25.0)/1000.0
        self.grid_size = 0.5
        self.grid_resolution = 20
        
        # read in first snapshot, get rotation matrix
        self.first_snapshot = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', startnum, 
                                                            simulation_directory=self.directory, assign_center=False)#,
                                                            #particle_subsample_factor=20)
        
        gizmo.io.Read.assign_center(self.first_snapshot)
        gizmo.io.Read.assign_principal_axes(self.first_snapshot)

        self.center_position = self.first_snapshot.center_position
        self.center_velocity = self.first_snapshot.center_velocity
        self.principal_axes_vectors = self.first_snapshot.principal_axes_vectors

        # store some other relevant information
        self.first_snapshot_time_in_Myr = self.first_snapshot.snapshot['time'] * 1000.0

        # read in last snapshot stars
        #self.last_snapshot_stars = gizmo.io.Read.read_snapshots('star', 'index', endnum, 
        #                                            simulation_directory=self.directory, assign_center=False)#,
        #                                            #particle_subsample_factor=20)

        # recenter first snapshot, last snapshot
        self._recenter_snapshot_(self.first_snapshot)
        #self._recenter_snapshot_(self.last_snapshot_stars)

        # read in all snapshots, but only the necessary quantities, and recenter
        self.snapshot_indices = range(startnum-num_prior, endnum+1)
        self.initial_key = num_prior

        self.snapshots = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.snapshot_indices, 
                                                        properties=['position', 'velocity', 'potential', 'id'], 
                                                        simulation_directory=self.directory, assign_center=False)#,
                                                        #particle_subsample_factor=20) #, properties=['position','potential'])
        for snap in self.snapshots:
            self._recenter_snapshot_(snap)

        # store some relevant data
        self.time_in_Myr = self._time_in_Myr_()

        # convert each snapshot to physical coordinates
        self._comoving_to_physical_(self.first_snapshot)
        for snap in self.snapshots:
            self._comoving_to_physical_(snap)
        
        # find starting star
        self.chosen_position_z0, self.chosen_velocity_z0, self.chosen_index_z0, self.chosen_id = self.starting_star(Rmin, Rmax, zmin, zmax)

        # set up trackers
        self._init_interpolators_()

        # set up grid
        self.grid = grid(self.grid_size, self.grid_resolution)
        self.grid.update_evolved_grid(self.chosen_positions[self.initial_key], self.chosen_velocities[self.initial_key])

        # set up rbfi for potential at each snapshot
        self._init_grid_rbfi_()
        self._execute_grid_rbfi_()

        # set up individual potential interpolators
        self._gen_potential_grid_interpolators_()
        self._execute_potential_grid_interpolators_(0.0)

    def _time_in_Myr_(self):
        original_times_in_Gyr = np.array([self.snapshots[i].snapshot['time'] for i in range(len(self.snapshots))])
        time_in_Myr = ( original_times_in_Gyr - original_times_in_Gyr[self.initial_key] ) * 1000.0
        return time_in_Myr

    def _potential_array_(self):
        all_star_potential = np.array([self.snapshots[i]['star']['potential'] for i in range(len(self.snapshots))])
        all_dark_potential = np.array([self.snapshots[i]['dark']['potential'] for i in range(len(self.snapshots))])
        out_star = np.zeros(np.shape(self.snapshots[0]['star']['potential'])).tolist()
        out_dark = np.zeros(np.shape(self.snapshots[0]['dark']['potential'])).tolist()
        for i in range(len(out_star)):
            out_star[i]= [all_star_potential[k][i] for k in range(len(self.snapshots))]
        for i in range(len(out_dark)):
            out_dark[i] = [all_dark_potential[k][i] for k in range(len(self.snapshots))]
        return np.concatenate((out_star,out_dark))

    def _position_array_(self):
        all_star_position = np.array([self.snapshots[i]['star']['position'] for i in range(len(self.snapshots))])
        all_dark_position = np.array([self.snapshots[i]['dark']['position'] for i in range(len(self.snapshots))])
        out_star = np.zeros(np.shape(self.snapshots[0]['star']['position'])).tolist()
        out_dark = np.zeros(np.shape(self.snapshots[0]['dark']['position'])).tolist()
        for i in range(len(out_star)):
            for j in range(3):
                out_star[i][j] = [all_star_position[k][i][j] for k in range(len(self.snapshots))]
        for i in range(len(out_dark)):
            for j in range(3):
                out_dark[i][j] = [all_dark_position[k][i][j] for k in range(len(self.snapshots))]
        return np.concatenate((out_star,out_dark))

    def _gen_interpolator_(self, pos_or_vel):
        interpolators = np.zeros(3).tolist()
        for i in range(3):
            interpolators[i] = interpolate.splrep(self.time_in_Myr, pos_or_vel[:,i])
            #interpolators[i][j] = interpolate.splrep(self.time_in_Myr, self.position_array[i][j], k=1)
        return interpolators
    
    def _init_interpolators_(self):
        self.chosen_indices = [int(np.where(self.snapshots[i]['star']['id'] == self.chosen_id)[0]) for i in range(len(self.snapshots)) ]
        self.chosen_positions = [self.snapshots[i]['star']['position'][self.chosen_indices[i]] for i in range(len(self.snapshots))]
        self.chosen_velocities = [self.snapshots[i]['star']['velocity'][self.chosen_indices[i]] for i in range(len(self.snapshots))]

        self.chosen_indices = np.array(self.chosen_indices)
        self.chosen_positions = np.array(self.chosen_positions)
        self.chosen_velocities = np.array(self.chosen_velocities)

        self.chosen_pos_interp = self._gen_interpolator_(self.chosen_positions)
        self.chosen_vel_interp = self._gen_interpolator_(self.chosen_velocities)

    def _gen_potential_grid_interpolators_(self):
        self.grid.grid_interpolators = []
        for i in range(len(self.grid.evolved_grid)):
            self.grid.grid_interpolators.append(interpolate.splrep(self.time_in_Myr, self.grid.snapshot_potential[:,i]))

    def _execute_potential_grid_interpolators_(self, t):
        self.grid.evolved_potential = []
        for i in range(len(self.grid.evolved_grid)):
            self.grid.evolved_potential.append(interpolate.splev(t, self.grid.grid_interpolators[i]))
        self.grid.evolved_potential = np.array(self.grid.evolved_potential)

    def _init_grid_rbfi_(self):
        self.grid_rbfi = []
        for i in range(len(self.snapshots)):
            all_positions = np.concatenate((self.snapshots[i]['star']['position'], self.snapshots[i]['dark']['position'], self.snapshots[i]['gas']['position']))
            all_potentials = np.concatenate((self.snapshots[i]['star']['potential'], self.snapshots[i]['dark']['potential'], self.snapshots[i]['gas']['potential']))
            rmag = np.linalg.norm(np.subtract(all_positions, self.chosen_positions[i]), axis=1)
            keys = np.where(rmag < self.grid_rmax)
            self.grid_rbfi.append(RBFInterpolant(all_positions[keys], all_potentials[keys], basis=self.basis, order=self.order))

    def _execute_grid_rbfi_(self):
        self.grid.snapshot_potential = []
        for i in range(len(self.snapshots)):
            self.grid.update_evolved_grid(self.chosen_positions[i], self.chosen_velocities[i])
            self.grid.snapshot_potential.append(self.grid_rbfi[i](self.grid.evolved_grid))
        self.grid.snapshot_potential = np.array(self.grid.snapshot_potential)

    def _init_potential_interpolators_(self):
        interpolators = np.zeros(np.shape(self.potential_array)[0:2]).tolist()
        for i in range(len(interpolators)):
            interpolators[i] = interpolate.splrep(self.time_in_Myr, self.potential_array[i])
            #interpolators[i] = interpolate.splrep(self.time_in_Myr, self.potential_array[i], k=1)
        return interpolators
    
    def _recenter_snapshot_(self, part):
        if part.snapshot['index'] == self.startnum:
            this_center_position = self.center_position
        else:
            snapshot_time_in_Myr = part.snapshot['time'] * 1000.0
            offset = self.center_velocity * (snapshot_time_in_Myr - self.first_snapshot_time_in_Myr)
            offset *= self.convert_kms_Myr_to_kpc
            this_center_position = self.center_position + offset

        print 'this center position:', this_center_position

        for key in part.keys():
            part[key]['position'] = np.subtract(part[key]['position'],this_center_position)
            part[key]['position'] = np.transpose(np.tensordot(self.principal_axes_vectors, np.transpose(part[key]['position']), axes=1))
            if 'velocity' in part[key].keys():
                part[key]['velocity'] = np.subtract(part[key]['velocity'],self.center_velocity)

    def _comoving_to_physical_(self, part):
        for key in part.keys():
            part[key]['position'] *= part.snapshot['scalefactor']
    
    def _clean_central_particles_(self):
        # TODO REWRITE THIS
        rmag_star = np.linalg.norm(self.first_snapshot['star']['position'], axis=1)
        rmag_dark = np.linalg.norm(self.first_snapshot['dark']['position'], axis=1)
        rmag = np.concatenate((rmag_star, rmag_dark))
        rbool = np.logical_and(rmag > self.rcut_min, rmag < self.rcut_max)
        keys = np.where(rbool)[0]
        self.position_array = self.position_array[keys]
        self.potential_array = self.potential_array[keys]

    def evolve_model(self, time, timestep=None):

        this_t_in_Myr = time.value_in(units.Myr)
        self.chosen_evolved_position = [float(interpolate.splev(this_t_in_Myr, self.chosen_pos_interp[i])) for i in range(3)]
        self.chosen_evolved_velocity = [float(interpolate.splev(this_t_in_Myr, self.chosen_vel_interp[i])) for i in range(3)]

        self.grid.update_evolved_grid(self.chosen_evolved_position, self.chosen_evolved_velocity)
        self._execute_potential_grid_interpolators_(this_t_in_Myr)
        
        self.grid._grid_evolved_kdtree_ = cKDTree(self.grid.evolved_grid)

    def _get_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self.grid._grid_evolved_kdtree_.query([x,y,z], self.nclose)
        rbfi = RBFInterpolant(self.grid.evolved_grid[ids], self.grid.evolved_potential[ids], basis=self.basis, order=self.order)
        return rbfi
    
    def get_potential_at_point(self, eps, xlist, ylist, zlist):
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            potlist = []
            for x,y,z in zip(xlist,ylist,zlist):
                rbfi = self._get_rbfi_(x,y,z)
                potlist.append( rbfi( [[x,y,z],[x,y,z]] )[0] )
            #return potlist | units.kms * units.kms
            return potlist
        
        else:
            rbfi = self._get_rbfi_(xlist, ylist, zlist)
        return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0]
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        #return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0] | units.kms * units.kms

    def get_gravity_at_point(self, eps, xlist, ylist, zlist):
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            axlist = []
            aylist = []
            azlist = []
            for x,y,z in zip(xlist,ylist,zlist):
                rbfi = self._get_rbfi_(x,y,z)
                axlist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(1,0,0) )[0] )
                aylist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(0,1,0) )[0] )
                azlist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(0,0,1) )[0] )
            # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
            ax = axlist# | units.kms*units.kms/units.kpc
            ay = aylist# | units.kms*units.kms/units.kpc
            az = azlist# | units.kms*units.kms/units.kpc
            return ax, ay, az 
        
        else:
            rbfi = self._get_rbfi_(xlist, ylist, zlist)
            # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
            ax = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(1,0,0) )[0]# | units.kms*units.kms/units.kpc
            ay = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(0,1,0) )[0]# | units.kms*units.kms/units.kpc
            az = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(0,0,1) )[0]# | units.kms*units.kms/units.kpc
            return ax, ay, az


    def starting_star(self, Rmin, Rmax, zmin, zmax, agemin_in_Gyr=1.2, agemax_in_Gyr=2.0, seed=1776):
        np.random.seed(seed)
        #starages = self.first_snapshot['star'].prop('age')
        #pos = self.first_snapshot['star']['position']
        #vel = self.first_snapshot['star']['velocity']
        
        starages = self.first_snapshot['star'].prop('age')
        pos = self.first_snapshot['star']['position']
        vel = self.first_snapshot['star']['velocity']

        Rstar = np.sqrt(pos[:,0]*pos[:,0] + pos[:,1]*pos[:,1])
        zstar = pos[:,2]

        agebool = np.logical_and(starages > agemin_in_Gyr, starages < agemax_in_Gyr)
        Rbool = np.logical_and(Rstar > Rmin, Rstar < Rmax)
        zbool = np.logical_and(zstar > zmin, zstar < zmax)

        totbool = np.logical_and(np.logical_and(agebool,Rbool), zbool)
        keys = np.where(totbool)[0]

        chosen_one = np.random.choice(keys)
        chosen_id = self.first_snapshot['star']['id'][chosen_one]
        return pos[chosen_one], vel[chosen_one], chosen_one, chosen_id


if __name__ == '__main__':
    import sys
    simulation_directory = sys.argv[1]
    startnum = int(sys.argv[2])
    endnum = int(sys.argv[3])
    g = gizmo_interface(simulation_directory, startnum, endnum)
