import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import gizmo_analysis as gizmo
import utilities as ut
from scipy import interpolate
from scipy.spatial import cKDTree
from rbf.interpolate import RBFInterpolant
from rbf.basis import phs3
from grid import grid
from amuse.units import units
from options import options_reader
import cPickle as pickle
from hashlib import md5
import time

class gizmo_interface(object):
    def __init__(self, options_file):
        
        self.convert_kms_Myr_to_kpc = 20000.0*np.pi / (61478577.0) # thanks wolfram alpha
        opt = options_reader(options_file)
        opt.set_options(self)

        self._read_snapshots_()            
        
        # find starting star
        self._init_starting_star_()
        
        # set up trackers
        self._init_starting_star_interpolators_()

        # set up grid
        self._init_grid_()

    def _read_snapshots_(self):
        # read in first snapshot, get rotation matrix
        
        """
        head = gizmo.io.Read.read_header(snapshot_value=self.startnum, simulation_directory=self.simulation_directory)
        sim_name = head['simulation.name'].replace(" ","_")
        cache_name = self.cache_directory + '/first_snapshot_'+sim_name+'_index'+str(self.startnum)+'.p'
        
        try:
            self.first_snapshot = dill.load(open(cache_name, 'rb'))
        except:
            self.first_snapshot = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.startnum, 
                                                    simulation_directory=self.simulation_directory, assign_center=False)#,
                                                    #particle_subsample_factor=20)
            dill.dump(self.first_snapshot, open(cache_name, 'wb'), protocol=2)
        """

        self.first_snapshot = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.startnum, 
                                        simulation_directory=self.simulation_directory, assign_center=False)#,
                                        #particle_subsample_factor=20)

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

        """
        init = self.snapshot_indices[0]
        fin = self.snapshot_indices[-1]
        cache_name = self.cache_directory + '/snapshots_'+sim_name+'_start'+str(init)+'_end'+str(fin)+'.p'
        
        try:
            self.snapshots = dill.load(open(cache_name, 'rb'))
        except:
            self.snapshots = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.snapshot_indices, 
                                                        properties=['position', 'velocity', 'potential', 'id'], 
                                                        simulation_directory=self.simulation_directory, assign_center=False)#,
                                                        #particle_subsample_factor=20) #, properties=['position','potential'])
            dill.dump(self.snapshots, open(cache_name, 'wb'), protocol=2)
        """

        self.snapshots = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', self.snapshot_indices, 
                                            properties=['position', 'velocity', 'potential', 'id'], 
                                            simulation_directory=self.simulation_directory, assign_center=False)#,
                                            #particle_subsample_factor=20) #, properties=['position','potential'])

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

        print 'this center position:', this_center_position
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

    def _grid_cache_name_(self):
        cache_name = 'grid_'
        cache_name += self.first_snapshot.info['simulation.name'].replace(' ','_')
        cache_name += '_ssid' + str(self.chosen_id)
        cache_name += '_x'+str(self.grid_x_size) + '_y' + str(self.grid_y_size)
        cache_name += '_z' + str(self.grid_z_size) + '_buffer' + str(self.grid_buffer)
        cache_name += '_res' + str(self.grid_resolution) + '_start'+str(self.startnum)
        cache_name += '_end'+str(self.endnum)+'_numprior'+str(self.num_prior)
        cache_name += '_nclose'+str(self.nclose)+'_basis'+str(self.basis)+'_order'+str(self.order)
        return cache_name, self.cache_directory + '/' + cache_name

    def _init_grid_(self):
        cache_name, cache_file = self._grid_cache_name_()
        try:
            self.grid = pickle.load(open(cache_file, 'rb'))
            return None
        except:
            print 'couldnt find cached grid:'
            print cache_name
            print 'constructing...'
            pass

        self.grid = grid(self.grid_x_size, self.grid_y_size, self.grid_z_size, self.grid_resolution,
                            self.ss_Rguess)
        self.grid.snapshot_potential = []

        for i in range(len(self.snapshots)):
            position = self.chosen_snapshot_positions[i]
            velocity = self.chosen_snapshot_velocities[i]
            snap = self.snapshots[i]

            self.grid.update_evolved_grid(position, velocity)

            self.grid.snapshot_potential.append(
                self._populate_grid_potential_(snap, self.grid, self.grid_x_max, 
                                                self.grid_y_max, self.grid_z_max))
        
        self.grid.snapshot_potential = np.array(self.grid.snapshot_potential)

        self._init_potential_grid_interpolators_()

        pickle.dump(self.grid, open(cache_file, 'wb'), protocol=2)
    
    def _populate_grid_potential_(self, snap, grid, x_max, y_max, z_max):
        position = grid.evolved_position
        velocity = grid.evolved_velocity

        # convert position to be at ss_Rguess
        position[2] = 0.0
        position /= np.linalg.norm(position)
        position *= self.ss_Rguess

        all_positions = np.concatenate((snap['star'].prop('host.distance.principal'), 
                snap['dark'].prop('host.distance.principal'),
                snap['gas'].prop('host.distance.principal')))
        all_potentials = np.concatenate((snap['star']['potential'], 
                snap['dark']['potential'], snap['gas']['potential']))

        rel_positions = np.subtract(all_positions, position)
        x_mag = np.abs(rel_positions[:,0])
        y_mag = np.abs(rel_positions[:,1])
        z_mag = np.abs(rel_positions[:,2])

        xbool = x_mag < x_max
        ybool = y_mag < y_max
        zbool = z_mag < z_max
        keys = np.where(np.logical_and(np.logical_and(xbool,ybool), zbool))[0]
        print 'key length:', len(keys)

        self._snapshot_relevant_positions_ = all_positions[keys]
        self._snapshot_relevant_potentials_ = all_potentials[keys]
        self._snapshot_relevant_kdtree_ = cKDTree(self._snapshot_relevant_positions_)


        return self._get_grid_potential_at_point_(0, grid.evolved_grid[:,0],
                        grid.evolved_grid[:,1], grid.evolved_grid[:,2])


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

    def _get_grid_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self._snapshot_relevant_kdtree_.query([x,y,z], self.nclose)
        rbfi = RBFInterpolant(self._snapshot_relevant_positions_[ids], self._snapshot_relevant_potentials_[ids], 
                            basis=self.basis, order=self.order)
        return rbfi

    def _get_grid_potential_at_point_(self, eps, xlist, ylist, zlist):
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        # xlist = xlist.value_in(units.kpc)
        # ylist = ylist.value_in(units.kpc)
        # zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            potlist = []
            for x,y,z in zip(xlist,ylist,zlist):
                #rbfi = self._get_rbfi_(x,y,z)
                rbfi = self._get_grid_rbfi_(x, y, z)
                potlist.append( rbfi( [[x,y,z],[x,y,z]] )[0] )
            # return potlist | units.kms * units.kms
            return potlist
        
        else:
            #rbfi = self._get_rbfi_(xlist, ylist, zlist)
            rbfi = self._get_grid_rbfi_(x, y, z)
        return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0]
        # return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0] | units.kms * units.kms



    """
    def _init_grid_rbfi_(self):
        self.grid_rbfi = []
        for i in range(len(self.snapshots)):
            all_positions = np.concatenate((self.snapshots[i]['star']['position'], 
                self.snapshots[i]['dark']['position'], self.snapshots[i]['gas']['position']))
            all_potentials = np.concatenate((self.snapshots[i]['star']['potential'], 
                self.snapshots[i]['dark']['potential'], self.snapshots[i]['gas']['potential']))
            
            #xy_mag = np.linalg.norm(np.subtract(all_positions, self.chosen_snapshot_positions[i]), axis=1)
            
            rel_positions = np.subtract(all_positions, self.chosen_snapshot_positions[i])
            xy_mag = np.maximum(np.abs(rel_positions[:,0]), np.abs(rel_positions[:,1]))
            z_mag = np.abs(all_positions[:,2])

            xybool = xy_mag < self.grid_xy_max
            zbool = z_mag < self.grid_z_max
            keys = np.where(np.logical_and(xybool, zbool))[0]
            print 'key length:', len(keys)

            start = time.time()
            self.grid_rbfi.append(RBFInterpolant(all_positions[keys], all_potentials[keys], basis=self.basis, order=self.order))
            end = time.time()
            print 'time (s):', end-start
    """

    def _execute_grid_rbfi_(self):
        self.grid.snapshot_potential = []
        for i in range(len(self.snapshots)):
            self.grid.update_evolved_grid(self.chosen_snapshot_positions[i], self.chosen_snapshot_velocities[i])



            #self.grid.snapshot_potential.append(self.grid_rbfi[i](self.grid.evolved_grid))
            xlist = self.grid.evolved_grid[:,0]
            ylist = self.grid.evolved_grid[:,1]
            zlist = self.grid.evolved_grid[:,2]
            self.grid.snapshot_potential.append(self.get_potential_at_point(0 | units.kpc,
                                                    xlist | units.kpc, ylist | units.kpc, zlist | units.kpc))
        self.grid.snapshot_potential = np.array(self.grid.snapshot_potential)

    def _init_potential_interpolators_(self):
        interpolators = np.zeros(np.shape(self.potential_array)[0:2]).tolist()
        for i in range(len(interpolators)):
            interpolators[i] = interpolate.splrep(self.time_in_Myr, self.potential_array[i])
            #interpolators[i] = interpolate.splrep(self.time_in_Myr, self.potential_array[i], k=1)
        return interpolators
    

    
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
        self._evolve_starting_star_(this_t_in_Myr)

        
        self.grid.update_evolved_grid(self.chosen_evolved_position, self.chosen_evolved_velocity)
        self._execute_potential_grid_interpolators_(this_t_in_Myr)
        
        self.grid._grid_evolved_kdtree_ = cKDTree(self.grid.evolved_grid)
        self._evolved_rbfi_ = RBFInterpolant(self.grid.evolved_grid, self.grid.evolved_potential, basis=self.basis, order=self.order)
        
        print 'evolved model to t (Myr):', this_t_in_Myr

    def _evolve_starting_star_(self, time_in_Myr):
        self.chosen_evolved_position = [float(interpolate.splev(time_in_Myr, self.chosen_pos_interp[i])) for i in range(3)]
        self.chosen_evolved_velocity = [float(interpolate.splev(time_in_Myr, self.chosen_vel_interp[i])) for i in range(3)]

    def _get_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self.grid._grid_evolved_kdtree_.query([x,y,z], self.nclose)
        rbfi = RBFInterpolant(self.grid.evolved_grid[ids], self.grid.evolved_potential[ids], basis=self.basis, order=self.order)
        return rbfi
    
    """
    This is one rbfi interpolator for each grid...
    def get_potential_at_point(self, eps, xlist, ylist, zlist):
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            positions = np.transpose([xlist, ylist, zlist])
            rbfi = self._evolved_rbfi_
            potlist = rbfi( positions )
            return potlist | units.kms * units.kms
        
        else:
            rbfi = self._evolved_rbfi_
        #return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0]
        return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0] | units.kms * units.kms
    """

    def get_potential_at_point(self, eps, xlist, ylist, zlist):
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            potlist = []
            for x,y,z in zip(xlist,ylist,zlist):
                #rbfi = self._get_rbfi_(x,y,z)
                rbfi = self._evolved_rbfi_
                potlist.append( rbfi( [[x,y,z],[x,y,z]] )[0] )
            return potlist | units.kms * units.kms
            #return potlist
        
        else:
            #rbfi = self._get_rbfi_(xlist, ylist, zlist)
            rbfi = self._evolved_rbfi_
        #return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0]
        return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0] | units.kms * units.kms

    
    def get_gravity_at_point(self, eps, xlist, ylist, zlist):
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        rbfi = self._evolved_rbfi_

        if hasattr(xlist,'__iter__'):
            positions = np.transpose([xlist, ylist, zlist])
            axlist = -rbfi(positions, diff=(1,0,0))
            aylist = -rbfi(positions, diff=(0,1,0))
            azlist = -rbfi(positions, diff=(0,0,1))
            ax = axlist | units.kms*units.kms/units.kpc
            ay = aylist | units.kms*units.kms/units.kpc
            az = azlist | units.kms*units.kms/units.kpc
            return ax, ay, az 
        
        else:
            ax = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(1,0,0) )[0] | units.kms*units.kms/units.kpc
            ay = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(0,1,0) )[0] | units.kms*units.kms/units.kpc
            az = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(0,0,1) )[0] | units.kms*units.kms/units.kpc
            return ax, ay, az
    
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
    """

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
    import sys
    options_file = sys.argv[1]
    g = gizmo_interface(options_file)
