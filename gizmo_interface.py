import matplotlib; matplotlib.use('agg')

import numpy as np
import gizmo_analysis as gizmo
import utilities as ut
from scipy import interpolate
from scipy.spatial import cKDTree
from rbf.interpolate import RBFInterpolant
from rbf.basis import phs3
from oceanic.grid_cartesian import grid
from amuse.units import units
from oceanic.options import options_reader
import pickle
import time

from tqdm import tqdm

from joblib import Parallel, delayed

import sys

class gizmo_interface(object):
    def __init__(self, options_reader, grid_snapshot=None):
        
        self.convert_kms_Myr_to_kpc = 20000.0*np.pi / (61478577.0) # thanks wolfram alpha
        self.kpc_in_km = 3.08567758E16
        #opt = options_reader(options_file)
        options_reader.set_options(self)

        self._read_snapshots_()            
        
        # find starting star
        self._init_starting_star_()
        
        # set up trackers
        self._init_starting_star_interpolators_()

        # set up grid
        if grid_snapshot is not None:
            self._init_grid_(grid_snapshot)
        else:    
            self._init_grid_()

        self.evolve_model(0 | units.Myr)

    def _read_snapshots_(self):
        # read in first snapshot, get rotation matrix
        
        # just gonna take a peak into the sim and see if we have it in cache
        head = gizmo.io.Read.read_header(snapshot_value=self.startnum, simulation_directory=self.simulation_directory)
        if self.sim_name is None:
            self.sim_name = head['simulation.name'].replace(" ","_")
        cache_name = 'first_snapshot_'+self.sim_name+'_index'+str(self.startnum)+'.p'
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
        self.first_snapshot = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.startnum, 
                                        simulation_directory=self.simulation_directory, assign_center=False)#,
                                        #particle_subsample_factor=20)
        """

        gizmo.io.Read.assign_center(self.first_snapshot)
        gizmo.io.Read.assign_principal_axes(self.first_snapshot)
        self.center_position = self.first_snapshot.center_position
        self.center_velocity = self.first_snapshot.center_velocity
        self.principal_axes_vectors = self.first_snapshot.principal_axes_vectors
        
        # store some other relevant information
        self.first_snapshot_time_in_Myr = self.first_snapshot.snapshot['time'] * 1000.0
        
        # read in all snapshots, but only the necessary quantities, and recenter
        self.snapshot_indices = range(self.startnum-self.num_prior, self.endnum+1)
        self.initial_key = self.num_prior

        init = self.snapshot_indices[0]
        fin = self.snapshot_indices[-1]
        cache_name = 'snapshots_'+self.sim_name+'_start'+str(init)+'_end'+str(fin)+'.p'
        cache_file = self.cache_directory + '/' + cache_name

        try:
            self.snapshots = pickle.load(open(cache_file, 'rb'))
            print('found and loaded cached file for snapshots:')
            print(cache_name)
        except:
            print('couldnt find cached file for snapshots:')
            print(cache_name)
            print('constructing...')
            self.snapshots = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.snapshot_indices, 
                                                        properties=['position', 'velocity', 'acceleration', 'id'], 
                                                        simulation_directory=self.simulation_directory, assign_center=False)#,
                                                        #particle_subsample_factor=20) #, properties=['position','potential'])
            pickle.dump(self.snapshots, open(cache_file, 'wb'), protocol=4)

        """
        self.snapshots = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.snapshot_indices, 
                                            properties=['position', 'velocity', 'potential', 'id'], 
                                            simulation_directory=self.simulation_directory, assign_center=False)#,
                                            #particle_subsample_factor=20) #, properties=['position','potential'])
        """

        for snap in self.snapshots:
            self._assign_self_center_(snap)

        # store some relevant data
        self.time_in_Myr = self._time_in_Myr_()

    def _assign_self_center_(self, part):
        if part.snapshot['index'] == self.startnum:
            this_center_position = self.center_position
        else:
            snapshot_time_in_Myr = part.snapshot['time'] * 1000.0
            offset = self.center_velocity * (snapshot_time_in_Myr - self.first_snapshot_time_in_Myr)
            offset *= self.convert_kms_Myr_to_kpc
            this_center_position = self.center_position + offset

        print('this center position:', this_center_position)
        part.center_position = this_center_position
        part.center_velocity = self.center_velocity
        part.principal_axes_vectors = self.principal_axes_vectors
        for key in part.keys():
            part[key].center_position = this_center_position
            part[key].center_velocity = self.center_velocity
            part[key].principal_axes_vectors = self.principal_axes_vectors

    def _time_in_Myr_(self):
        original_times_in_Gyr = np.array([self.snapshots[i].snapshot['time'] for i in range(len(self.snapshots))])
        time_in_Myr = ( original_times_in_Gyr - original_times_in_Gyr[self.initial_key] ) * 1000.0
        return time_in_Myr
    
    def _init_starting_star_interpolators_(self):
        self.chosen_indices = [int(np.where(self.snapshots[i]['star']['id'] == self.chosen_id)[0]) for i in range(len(self.snapshots)) ]
        self.chosen_snapshot_positions = [self.snapshots[i]['star'].prop('host.distance.principal')[self.chosen_indices[i]] for i in range(len(self.snapshots))]
        self.chosen_snapshot_velocities = [self.snapshots[i]['star'].prop('host.velocity.principal')[self.chosen_indices[i]] for i in range(len(self.snapshots))]

        self.chosen_indices = np.array(self.chosen_indices)
        self.chosen_snapshot_positions = np.array(self.chosen_snapshot_positions)
        self.chosen_snapshot_velocities = np.array(self.chosen_snapshot_velocities)

        self._set_ss_Rguess_(self.chosen_snapshot_positions)

        self.chosen_pos_interp = self._gen_pos_or_vel_interpolator_(self.chosen_snapshot_positions)
        self.chosen_vel_interp = self._gen_pos_or_vel_interpolator_(self.chosen_snapshot_velocities)

    def _set_ss_Rguess_(self, snap_positions):
        Rlist = np.sqrt(snap_positions[:,0]**2. + snap_positions[:,1]**2.)
        self.ss_Rguess = np.mean(Rlist)

    def _gen_pos_or_vel_interpolator_(self, pos_or_vel):
        interpolators = np.zeros(3).tolist()
        for i in range(3):
            interpolators[i] = interpolate.splrep(self.time_in_Myr, pos_or_vel[:,i])
            #interpolators[i][j] = interpolate.splrep(self.time_in_Myr, self.position_array[i][j], k=1)
        return interpolators

    def _grid_cache_name_(self, snapshot_index=None):
        if snapshot_index is not None:
            cache_name = 'grid_snapshot'+str(snapshot_index)+'_'
        else:
            cache_name = 'grid_'
        cache_name += self.sim_name
        cache_name += '_ssid' + str(self.chosen_id)
        cache_name += '_gridseed' + str(self.grid_seed)
        cache_name += '_Rmin'+str(self.grid_R_min) + '_Rmax' + str(self.grid_R_max)
        cache_name += '_zcut' + str(self.grid_z_max) + '_phi' + str(self.grid_phi_size)
        cache_name += '_N' + str(self.grid_N) + '_start'+str(self.startnum)
        cache_name += '_end'+str(self.endnum)+'_numprior'+str(self.num_prior)
        cache_name += '_nclose'+str(self.nclose)+'_basis'
        cache_name += str(self.basis).replace(' ','').replace('*','').replace(':','')
        cache_name += '_order'+str(self.order)
        return cache_name, self.cache_directory + '/' + cache_name

    """
    old potential init grid
    def _init_grid_(self, grid_snapshot=None):
        grid_cache_name, grid_cache_file = self._grid_cache_name_()
        try:
            self.grid = pickle.load(open(cache_file, 'rb'))
            return None
        except:
            print('couldnt find cached grid:')
            print(grid_cache_name)
            print('constructing...')
            pass

        cyl_positions = np.concatenate((self.first_snapshot['star'].prop('host.distance.principal.cylindrical'), 
            self.first_snapshot['dark'].prop('host.distance.principal.cylindrical'),
            self.first_snapshot['gas'].prop('host.distance.principal.cylindrical')))

        self.grid = grid(self.grid_R_min, self.grid_R_max, self.grid_z_max,
                         self.grid_phi_size, self.grid_N, self.grid_seed, cyl_positions)
        self.grid.snapshot_potential = []

        # TODO clean up this section

        # if user specified to calc a snapshot's grid, then do so, dump to cache, and quit
        snap_indices = np.array([self.snapshots[i].snapshot['index'] for i in range(len(self.snapshots))])
        if grid_snapshot is not None:
            snap_cache_name, snap_cache_file = self._grid_cache_name_(grid_snapshot)
            print('generating snapshot grid for this file:')
            print(snap_cache_name)

            key = int(np.where(grid_snapshot == snap_indices)[0])
            position = self.chosen_snapshot_positions[key]
            velocity = self.chosen_snapshot_velocities[key]
            snap = self.snapshots[key]

            self._ss_phi_ = np.mod(np.arctan2(position[1],position[0]), 2.*np.pi)
            self.grid.update_evolved_grid(self._ss_phi_)

            this_snapshot_grid = self._populate_grid_potential_(snap, self.grid)
            pickle.dump(this_snapshot_grid, open(snap_cache_file, 'wb'), protocol=4)
            print('done, will quit now...')
            quit()

        for i in range(len(self.snapshots)):
            position = self.chosen_snapshot_positions[i]
            velocity = self.chosen_snapshot_velocities[i]
            snap = self.snapshots[i]

            self._ss_phi_ = np.mod(np.arctan2(position[1],position[0]), 2.*np.pi)
            self.grid.update_evolved_grid(self._ss_phi_)

            try:
                snap_cache_name, snap_cache_file = self._grid_cache_name_(snap.snapshot['index'])
                this_snapshot_grid = pickle.load(open(snap_cache_file, 'rb'))
            except:
                this_snapshot_grid = self._populate_grid_potential_(snap, self.grid)
                pickle.dump(this_snapshot_grid, open(snap_cache_file, 'wb'), protocol=4)

            self.grid.snapshot_potential.append(
                self._populate_grid_potential_(snap, self.grid))
        
        self.grid.snapshot_potential = np.array(self.grid.snapshot_potential)

        self._init_potential_grid_interpolators_()

        pickle.dump(self.grid, open(grid_cache_file, 'wb'), protocol=4)
    """

    def _init_grid_(self, grid_snapshot=None):
        grid_cache_name, grid_cache_file = self._grid_cache_name_()
        try:
            self.grid = pickle.load(open(grid_cache_file, 'rb'))
            return None
        except:
            print('couldnt find cached grid:')
            print(grid_cache_name)
            print('constructing...')
            pass

        cyl_positions = np.concatenate((self.first_snapshot['star'].prop('host.distance.principal.cylindrical'), 
            self.first_snapshot['dark'].prop('host.distance.principal.cylindrical'),
            self.first_snapshot['gas'].prop('host.distance.principal.cylindrical')))

        self.grid = grid(self.grid_R_min, self.grid_R_max, self.grid_z_max,
                         self.grid_phi_size, self.grid_N, self.grid_seed, cyl_positions)
        self.grid.snapshot_acceleration_x = []
        self.grid.snapshot_acceleration_y = []
        self.grid.snapshot_acceleration_z = []

        # TODO clean up this section

        # if user specified to calc a snapshot's grid, then do so, dump to cache, and quit
        snap_indices = np.array([self.snapshots[i].snapshot['index'] for i in range(len(self.snapshots))])
        if grid_snapshot is not None:
            
            snap_cache_name, snap_cache_file = self._grid_cache_name_(grid_snapshot)
            snap_cache_file_x = snap_cache_file.replace('snapshot', 'snapshot_x')
            snap_cache_file_y = snap_cache_file.replace('snapshot', 'snapshot_y')
            snap_cache_file_z = snap_cache_file.replace('snapshot', 'snapshot_z')
            
            print('generating snapshot grid for this file:')
            print(snap_cache_name)

            key = int(np.where(grid_snapshot == snap_indices)[0])
            position = self.chosen_snapshot_positions[key]
            velocity = self.chosen_snapshot_velocities[key]
            snap = self.snapshots[key]

            self._ss_phi_ = np.mod(np.arctan2(position[1],position[0]), 2.*np.pi)
            self.grid.update_evolved_grid(self._ss_phi_)

            this_snapshot_grid_x, this_snapshot_grid_y, this_snapshot_grid_z = \
                self._populate_grid_acceleration_(snap, self.grid)
            
            pickle.dump(this_snapshot_grid_x, open(snap_cache_file_x, 'wb'), protocol=4)
            pickle.dump(this_snapshot_grid_y, open(snap_cache_file_y, 'wb'), protocol=4)
            pickle.dump(this_snapshot_grid_z, open(snap_cache_file_z, 'wb'), protocol=4)
            print('done, will quit now...')
            sys.exit(0)

        for i in range(len(self.snapshots)):
            position = self.chosen_snapshot_positions[i]
            velocity = self.chosen_snapshot_velocities[i]
            snap = self.snapshots[i]

            self._ss_phi_ = np.mod(np.arctan2(position[1],position[0]), 2.*np.pi)
            self.grid.update_evolved_grid(self._ss_phi_)

            try:
                snap_cache_name, snap_cache_file = self._grid_cache_name_(snap.snapshot['index'])
                snap_cache_file_x = snap_cache_file.replace('snapshot', 'snapshot_x')
                snap_cache_file_y = snap_cache_file.replace('snapshot', 'snapshot_y')
                snap_cache_file_z = snap_cache_file.replace('snapshot', 'snapshot_z')
                
                this_snapshot_grid_x = pickle.load(open(snap_cache_file_x, 'rb'))
                this_snapshot_grid_y = pickle.load(open(snap_cache_file_y, 'rb'))
                this_snapshot_grid_z = pickle.load(open(snap_cache_file_z, 'rb'))
            except:
                this_snapshot_grid_x, this_snapshot_grid_y, this_snapshot_grid_z = \
                    self._populate_grid_acceleration_(snap, self.grid)
                pickle.dump(this_snapshot_grid_x, open(snap_cache_file_x, 'wb'), protocol=4)
                pickle.dump(this_snapshot_grid_y, open(snap_cache_file_y, 'wb'), protocol=4)
                pickle.dump(this_snapshot_grid_z, open(snap_cache_file_z, 'wb'), protocol=4)

            self.grid.snapshot_acceleration_x.append(
                this_snapshot_grid_x)
            self.grid.snapshot_acceleration_y.append(
                this_snapshot_grid_y)
            self.grid.snapshot_acceleration_z.append(
                this_snapshot_grid_z)
        
        self.grid.snapshot_acceleration_x = np.array(self.grid.snapshot_acceleration_x)
        self.grid.snapshot_acceleration_y = np.array(self.grid.snapshot_acceleration_y)
        self.grid.snapshot_acceleration_z = np.array(self.grid.snapshot_acceleration_z)

        self._init_acceleration_grid_interpolators_()

        pickle.dump(self.grid, open(grid_cache_file, 'wb'), protocol=4)

    def _populate_grid_potential_(self, snap, grid):

        all_positions = np.concatenate((snap['star'].prop('host.distance.principal'), 
                snap['dark'].prop('host.distance.principal'),
                snap['gas'].prop('host.distance.principal')))

        all_cyl_positions = np.concatenate((snap['star'].prop('host.distance.principal.cylindrical'), 
                snap['dark'].prop('host.distance.principal.cylindrical'),
                snap['gas'].prop('host.distance.principal.cylindrical')))
        all_potentials = np.concatenate((snap['star']['potential'], 
                snap['dark']['potential'], snap['gas']['potential']))

        all_potentials /= snap.snapshot['scalefactor']**2.0

        Rbool = np.logical_and(all_cyl_positions[:,0] > self.grid_R_min,
                               all_cyl_positions[:,0] < self.grid_R_max)
        zbool = np.abs(all_cyl_positions[:,1]) < self.grid_z_max

        relphi = np.subtract(all_cyl_positions[:,2],self._ss_phi_)
        phibool = np.abs(relphi) < self.grid_phi_size/2.0

        keys = np.where(np.logical_and(np.logical_and(Rbool,zbool), phibool))[0]
        print('key length:', len(keys))

        self._snapshot_relevant_positions_ = all_positions[keys]
        self._snapshot_relevant_potentials_ = all_potentials[keys]

        rbfi = RBFInterpolant(self._snapshot_relevant_positions_, self._snapshot_relevant_potentials_,
                                basis = self.basis, order = self.order)
        return rbfi(grid.evolved_grid)

    def _populate_grid_acceleration_(self, snap, grid):

        all_positions = np.concatenate((snap['star'].prop('host.distance.principal'), 
                snap['dark'].prop('host.distance.principal'),
                snap['gas'].prop('host.distance.principal')))

        all_cyl_positions = np.concatenate((snap['star'].prop('host.distance.principal.cylindrical'), 
                snap['dark'].prop('host.distance.principal.cylindrical'),
                snap['gas'].prop('host.distance.principal.cylindrical')))
        all_accelerations = np.concatenate((snap['star']['acceleration'], 
                snap['dark']['acceleration'], snap['gas']['acceleration']))

        acc_fac = snap.snapshot['scalefactor'] * self.kpc_in_km
        acc_fac *= snap.info['hubble']
        all_accelerations = np.divide(all_accelerations, acc_fac)

        all_accelerations = ut.coordinate.get_coordinates_rotated(
                        all_accelerations, snap.principal_axes_vectors)

        Rbool = np.logical_and(all_cyl_positions[:,0] > self.grid_R_min,
                               all_cyl_positions[:,0] < self.grid_R_max)
        zbool = np.abs(all_cyl_positions[:,1]) < self.grid_z_max

        relphi = np.subtract(all_cyl_positions[:,2],self._ss_phi_)
        phibool = np.abs(relphi) < self.grid_phi_size/2.0

        keys = np.where(np.logical_and(np.logical_and(Rbool,zbool), phibool))[0]
        print('key length:', len(keys))

        self._snapshot_relevant_positions_ = all_positions[keys]
        self._snapshot_relevant_accelerations_ = all_accelerations[keys]

        rbfi_x = RBFInterpolant(self._snapshot_relevant_positions_, 
                                self._snapshot_relevant_accelerations_[:,0],
                                basis = self.basis, order = self.order)
        rbfi_y = RBFInterpolant(self._snapshot_relevant_positions_, 
                                self._snapshot_relevant_accelerations_[:,1],
                                basis = self.basis, order = self.order)
        rbfi_z = RBFInterpolant(self._snapshot_relevant_positions_, 
                                self._snapshot_relevant_accelerations_[:,2],
                                basis = self.basis, order = self.order)
        return rbfi_x(grid.evolved_grid), rbfi_y(grid.evolved_grid), rbfi_z(grid.evolved_grid)

    def _init_potential_grid_interpolators_(self):
        self.grid.grid_pot_interpolators = []
        for i in range(len(self.grid.evolved_grid)):
            self.grid.grid_pot_interpolators.append(
                    interpolate.splrep(self.time_in_Myr, self.grid.snapshot_potential[:,i]))

    def _execute_potential_grid_interpolators_(self, t):
        self.grid.evolved_potential = []
        for i in range(len(self.grid.evolved_grid)):
            self.grid.evolved_potential.append(interpolate.splev(t, self.grid.grid_pot_interpolators[i]))
        self.grid.evolved_potential = np.array(self.grid.evolved_potential)

    def _init_acceleration_grid_interpolators_(self):
        self.grid.grid_accx_interpolators = []
        self.grid.grid_accy_interpolators = []
        self.grid.grid_accz_interpolators = []
        for i in range(len(self.grid.evolved_grid)):
            self.grid.grid_accx_interpolators.append(
                    interpolate.splrep(self.time_in_Myr, self.grid.snapshot_acceleration_x[:,i]))
            self.grid.grid_accy_interpolators.append(
                    interpolate.splrep(self.time_in_Myr, self.grid.snapshot_acceleration_y[:,i]))
            self.grid.grid_accz_interpolators.append(
                    interpolate.splrep(self.time_in_Myr, self.grid.snapshot_acceleration_z[:,i]))

    def _execute_acceleration_grid_interpolators_(self, t):
        self.grid.evolved_acceleration_x = []
        self.grid.evolved_acceleration_y = []
        self.grid.evolved_acceleration_z = []
        for i in range(len(self.grid.evolved_grid)):
            self.grid.evolved_acceleration_x.append(interpolate.splev(t, self.grid.grid_accx_interpolators[i]))
            self.grid.evolved_acceleration_y.append(interpolate.splev(t, self.grid.grid_accy_interpolators[i]))
            self.grid.evolved_acceleration_z.append(interpolate.splev(t, self.grid.grid_accz_interpolators[i]))
        self.grid.evolved_acceleration_x = np.array(self.grid.evolved_acceleration_x)
        self.grid.evolved_acceleration_y = np.array(self.grid.evolved_acceleration_y)
        self.grid.evolved_acceleration_z = np.array(self.grid.evolved_acceleration_z)

    """
    old potential evolve_model
    def evolve_model(self, time, timestep=None):

        this_t_in_Myr = time.value_in(units.Myr)
        self._evolve_starting_star_(this_t_in_Myr)

        position = self.chosen_evolved_position
        phi = np.mod(np.arctan2(position[1],position[0]), 2.*np.pi)

        self.grid.update_evolved_grid(phi)
        self._execute_potential_grid_interpolators_(this_t_in_Myr)
        
        self.grid._grid_evolved_kdtree_ = cKDTree(self.grid.evolved_grid)
        #self._evolved_rbfi_ = RBFInterpolant(self.grid.evolved_grid, self.grid.evolved_potential, basis=self.basis, order=self.order)
        
        print('evolved model to t (Myr):', this_t_in_Myr)
    """

    def evolve_model(self, time, timestep=None):

        this_t_in_Myr = time.value_in(units.Myr)
        self._evolve_starting_star_(this_t_in_Myr)

        position = self.chosen_evolved_position
        phi = np.mod(np.arctan2(position[1],position[0]), 2.*np.pi)

        self.grid.update_evolved_grid(phi)
        self._execute_acceleration_grid_interpolators_(this_t_in_Myr)
        
        self.grid._grid_evolved_kdtree_ = cKDTree(self.grid.evolved_grid)
        
        print('evolved model to t (Myr):', this_t_in_Myr)

    def _evolve_starting_star_(self, time_in_Myr):
        self.chosen_evolved_position = [float(interpolate.splev(time_in_Myr, self.chosen_pos_interp[i])) for i in range(3)]
        self.chosen_evolved_velocity = [float(interpolate.splev(time_in_Myr, self.chosen_vel_interp[i])) for i in range(3)]

        self.chosen_evolved_position = np.array(self.chosen_evolved_position)
        self.chosen_evolved_velocity = np.array(self.chosen_evolved_velocity)


    def _get_pot_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self.grid._grid_evolved_kdtree_.query([x,y,z], self.nclose)
        rbfi = RBFInterpolant(self.grid.evolved_grid[ids], self.grid.evolved_potential[ids], basis=self.basis, order=self.order)
        return rbfi

    def _get_acc_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self.grid._grid_evolved_kdtree_.query([x,y,z], self.nclose)
        rbfi_x = RBFInterpolant(self.grid.evolved_grid[ids], self.grid.evolved_acceleration_x[ids],\
            basis=self.basis, order=self.order)
        rbfi_y = RBFInterpolant(self.grid.evolved_grid[ids], self.grid.evolved_acceleration_y[ids],\
            basis=self.basis, order=self.order)
        rbfi_z = RBFInterpolant(self.grid.evolved_grid[ids], self.grid.evolved_acceleration_z[ids],\
            basis=self.basis, order=self.order)
        return rbfi_x, rbfi_y, rbfi_z

    """
    def get_potential_at_point(self, eps, xlist, ylist, zlist):
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            potlist = []
            for x,y,z in zip(xlist,ylist,zlist):
                rbfi = self._get_rbfi_(x,y,z)
                #rbfi = self._evolved_rbfi_
                potlist.append( rbfi( [[x,y,z],[x,y,z]] )[0] )
            return potlist | units.kms * units.kms
            #return potlist
        
        else:
            rbfi = self._get_rbfi_(xlist, ylist, zlist)
            #rbfi = self._evolved_rbfi_
        #return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0]
        return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0] | units.kms * units.kms
    """

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
                rbfi_x, rbfi_y, rbfi_z = self._get_acc_rbfi_(x,y,z)
                axlist.append( float(rbfi_x( [[x,y,z]] )) )
                aylist.append( float(rbfi_y( [[x,y,z]] )) )
                azlist.append( float(rbfi_z( [[x,y,z]] )) )
            # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
            ax = axlist | units.kms/units.s
            ay = aylist | units.kms/units.s
            az = azlist | units.kms/units.s
            return ax, ay, az
        
        else:
            rbfi_x, rbfi_y, rbfi_z = self._get_acc_rbfi_(xlist, ylist, zlist)
            # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
            ax = float(rbfi_x([[xlist, ylist, zlist]])) | units.kms/units.s
            ay = float(rbfi_y([[xlist, ylist, zlist]])) | units.kms/units.s
            az = float(rbfi_z([[xlist, ylist, zlist]])) | units.kms/units.s
            return ax, ay, az
    

    def _init_starting_star_(self):
        self.chosen_position_z0, self.chosen_velocity_z0, self.chosen_index_z0, self.chosen_id = self.starting_star(
                            self.ss_Rmin, self.ss_Rmax, self.ss_zmin, self.ss_zmax,
                            self.ss_agemin_in_Gyr, self.ss_agemax_in_Gyr, self.ss_seed)

    def starting_star(self, Rmin, Rmax, zmin, zmax, agemin_in_Gyr, agemax_in_Gyr, seed=1776):
        np.random.seed(seed)
        #starages = self.first_snapshot['star'].prop('age')
        #pos = self.first_snapshot['star']['position']
        #vel = self.first_snapshot['star']['velocity']
        
        starages = self.first_snapshot['star'].prop('age')
        pos = self.first_snapshot['star'].prop('host.distance.principal')
        vel = self.first_snapshot['star'].prop('host.velocity.principal')

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
    options_file = sys.argv[1]
    opt = options_reader(options_file)

    if len(sys.argv) == 3:
        snap_index = int(sys.argv[2])
        g = gizmo_interface(opt, snap_index)
    else:
        g = gizmo_interface(opt)
