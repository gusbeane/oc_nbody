import matplotlib; matplotlib.use('agg')

import numpy as np
import gizmo_analysis as gizmo
from scipy import interpolate
from scipy.spatial import cKDTree
from rbf.interpolate import RBFInterpolant
from oceanic.grid_cartesian import grid
from amuse.units import units
from oceanic.options import options_reader
from oceanic.analysis import agama_wrapper
from oceanic.oc_code import agama_interpolator
import pickle

from pykdgrav import ConstructKDTree, GetAccelParallel
from astropy.constants import G as G_astropy
import astropy.units as u

import agama

from multiprocessing import Pool

import sys
import os


class acc_wrapper(object):
    def __init__(self):
        pass

    def init_interp(self, interpolators_x, interpolators_y,
                    interpolators_z):
        self.interpolators_x = interpolators_x
        self.interpolators_y = interpolators_y
        self.interpolators_z = interpolators_z

    def x(self, t, i):
        return interpolate.splev(t, self.interpolators_x[i])

    def y(self, t, i):
        return interpolate.splev(t, self.interpolators_y[i])

    def z(self, t, i):
        return interpolate.splev(t, self.interpolators_z[i])


def init_worker(interpolators_x, interpolators_y, interpolators_z):
    global acc_wrap
    acc_wrap = acc_wrapper()
    acc_wrap.init_interp(interpolators_x, interpolators_y, interpolators_z)


def run_worker_x(t, i):
    return acc_wrap.x(t, i)


def run_worker_y(t, i):
    return acc_wrap.y(t, i)


def run_worker_z(t, i):
    return acc_wrap.z(t, i)


class gizmo_interface(object):
    def __init__(self, options_reader, grid_snapshot=None):

        agama.setUnits(mass=1, length=1, velocity=1)

        self.G = G_astropy.to_value(u.kpc**2 * u.km / (u.s * u.Myr * u.Msun))
        self.theta = 0.5

        self.convert_kms_Myr_to_kpc = 20000.0*np.pi / (61478577.0)
        self.kpc_in_km = 3.08567758E16
        # opt = options_reader(options_file)
        options_reader.set_options(self)
        self.options_reader = options_reader

        self.snapshot_indices = range(self.startnum-self.num_prior,
                                self.endnum+1)
        self.initial_key = self.num_prior

        if not os.path.isdir(self.cache_directory):
            os.makedirs(self.cache_directory)

        if self.axisymmetric:
            if self.axisymmetric_tevolve:
                self._read_snapshots_()
                self._gen_axisymmetric_(all_snaps=True)
                self.evolve_model(0 | units.Myr)
            else:
                self._read_snapshots_(first_only=True)
                self._gen_axisymmetric_()
            return None

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

        self.grid._grid_evolved_kdtree_ = cKDTree(self.grid.evolved_grid)

    def _pot_cache_file_(self, index):
        potential_cache_file = self.cache_directory + '/potential_id'+str(index)
        potential_cache_file += '_' + self.sim_name + '_pot'
        return potential_cache_file

    def _gen_axisymmetric_(self, all_snaps=False):
        agama.setUnits(mass=1, length=1, velocity=1)
        if all_snaps:
            pot_file_list = []
            for index in self.snapshot_indices:
                pot_file_list.append(self._pot_cache_file_(index))
        else:
            pot_file_list = (self._pot_cache_file_(self.startnum),)

        for i,file in enumerate(pot_file_list):
            try:
                self.pdark = agama.Potential(file=file+'_dark')
                self.pbar = agama.Potential(file=file+'_bar')
                self.potential = agama.Potential(self.pdark, self.pbar)
            except:
                star_position = self.snapshots[i]['star'].prop('host.distance.principal')
                gas_position = self.snapshots[i]['gas'].prop('host.distance.principal')
                dark_position = self.snapshots[i]['dark'].prop('host.distance.principal')

                star_mass = self.snapshots[i]['star']['mass']
                gas_mass = self.snapshots[i]['gas']['mass']
                dark_mass = self.snapshots[i]['dark']['mass']

                position = np.concatenate((star_position, gas_position))
                mass = np.concatenate((star_mass, gas_mass))

                #TODO make these user-controllable
                self.pdark = agama.Potential(type="Multipole",
                                            particles=(dark_position, dark_mass),
                                            symmetry='a', gridsizeR=20, lmax=2)
                self.pbar = agama.Potential(type="CylSpline",
                                            particles=(position, mass),
                                            symmetry='a', gridsizer=20, gridsizez=20,
                                            mmax=0, Rmin=0.2,
                                            Rmax=50, Zmin=0.02, Zmax=10)
                self.pdark.export(file+'_dark')
                self.pbar.export(file+'_bar')
                self.potential = agama.Potential(self.pdark, self.pbar)
                self.potential.export(file)

        if self.axisymmetric_tevolve:
            self._gen_agama_interpolator_()

        return None

    def _gen_agama_interpolator_(self):
        bar_fnames = [self._pot_cache_file_(idx)+'_bar' for idx in self.snapshot_indices]
        dark_fnames = [self._pot_cache_file_(idx)+'_dark' for idx in self.snapshot_indices]
        self.bar_agamaint = agama_interpolator(bar_fnames, self.time_in_Myr)
        self.dark_agamaint = agama_interpolator(dark_fnames, self.time_in_Myr)

    def _agama_potential_(self, t):
        bar_pot = agama.Potential(file=self.bar_agamaint(t))
        dark_pot = agama.Potential(file=self.dark_agamaint(t))
        return agama.Potential(bar_pot, dark_pot)

    def _read_snapshots_(self, first_only=False):
        # read in first snapshot, get rotation matrix

        # just gonna take a peak into the sim and see if we have it in cache
        head = gizmo.io.Read.read_header(snapshot_value=self.startnum,
                                         simulation_directory=
                                         self.simulation_directory)
        if self.sim_name is None:
            self.sim_name = head['simulation.name'].replace(" ", "_")
        cache_name = 'first_snapshot_' + self.sim_name+'_index' + \
            str(self.startnum)+'.p'
        cache_file = self.cache_directory + '/' + cache_name

        try:
            self.first_snapshot = pickle.load(open(cache_file, 'rb'))
            print('found and loaded cached file for first_snapshot:')
            print(cache_name)

        except:
            print('couldnt find cached file for first_snapshot:')
            print(cache_name)
            print('constructing...')
            self.first_snapshot =\
                gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                             'index', self.startnum,
                                             simulation_directory=
                                             self.simulation_directory,
                                             assign_center=False)
            pickle.dump(self.first_snapshot,
                        open(cache_file, 'wb'), protocol=4)

        gizmo.io.Read.assign_center(self.first_snapshot)
        gizmo.io.Read.assign_principal_axes(self.first_snapshot)
        self.center_position = self.first_snapshot.center_position
        self.center_velocity = self.first_snapshot.center_velocity
        self.principal_axes_vectors =\
            self.first_snapshot.principal_axes_vectors

        # store some other relevant information
        self.first_snapshot_time_in_Myr =\
            self.first_snapshot.snapshot['time'] * 1000.0

        if first_only:
            return None

        # read in all snapshots,
        # but only the necessary quantities, and recenter


        # # # # # # # # # # # # # # # # # # # # # # #
        #                                           #
        #           Read in snapshots               #
        #                                           #
        # # # # # # # # # # # # # # # # # # # # # # #

        init = self.snapshot_indices[0]
        fin = self.snapshot_indices[-1]
        cache_name = 'snapshots_' + self.sim_name + '_start' + str(init)
        cache_name += '_end' + str(fin) + '_first' + str(self.startnum)
        cache_name += '_Rmag' + str(self.Rmax) + '.p'
        cache_file = self.cache_directory + '/' + cache_name

        try:
            self.snapshots = pickle.load(open(cache_file, 'rb'))
            print('found and loaded cached file for snapshots:')
            print(cache_name)
        except:
            print('couldnt find cached file for snapshots:')
            print(cache_name)
            print('constructing...')
            self.snapshots =\
                gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                             'index', self.snapshot_indices,
                                             properties=['position', 'id',
                                                         'mass', 'velocity',
                                                         'smooth.length',
                                                         'form.scalefactor'],
                                             simulation_directory=
                                             self.simulation_directory,
                                             assign_principal_axes=True)

            for i in range(len(self.snapshots)):
                self.snapshots[i] = self._clean_Rmag_(self.snapshots[i])

            pickle.dump(self.snapshots, open(cache_file, 'wb'), protocol=4)

        # # # # # # # # # # # # # # # # # # # # # # #
        #                                           #
        # Read in velocities of only star particles #
        #                                           #
        # # # # # # # # # # # # # # # # # # # # # # #

        cache_name = 'star_vel_' + self.sim_name + '_start' + str(init)
        cache_name += '_end' + str(fin) + '_first' + str(self.startnum)
        cache_name += '_Rmag' + str(self.Rmax) + '.p'
        cache_file = self.cache_directory + '/' + cache_name

        try:
            self.star_snapshots = pickle.load(open(cache_file, 'rb'))
            print('found and loaded cached file for star velocities:')
            print(cache_name)
        except:
            print('couldnt find cached file for star velocities:')
            print(cache_name)
            print('constructing...')
            self.star_snapshots =\
                gizmo.io.Read.read_snapshots(['star'],
                                             'index', self.snapshot_indices,
                                             properties=['velocity', 'id',
                                                         'position', 'mass'],
                                             simulation_directory=
                                             self.simulation_directory,
                                             assign_center=False)

            for snap in self.star_snapshots:
                self._assign_self_center_(snap)

            pickle.dump(self.star_snapshots, open(cache_file, 'wb'), protocol=4)

        # store some relevant data
        self.time_in_Myr = self._time_in_Myr_()

    def _clean_Rmag_(self, snap):
        # cleans out all particles greater than Rmag from galactic center
        for key in snap.keys():
            rmag = snap[key].prop('host.distance.total')
            rmag_keys = np.where(rmag < self.Rmax)[0]
            for dict_key in snap[key].keys():
                snap[key][dict_key] = snap[key][dict_key][rmag_keys]
        return snap

    def _assign_self_center_(self, part):
        gizmo.io.Read.assign_center(part)

        # if part.snapshot['index'] == self.startnum:
        #     this_center_position = self.center_position
        # else:
        #     snapshot_time_in_Myr = part.snapshot['time'] * 1000.0 - \
        #                            self.first_snapshot_time_in_Myr

        #     offset = self.center_velocity * snapshot_time_in_Myr
        #     offset *= self.convert_kms_Myr_to_kpc

        #     this_center_position = self.center_position + offset

        # print('this center position:', this_center_position)
        # part.center_position = this_center_position
        # part.center_velocity = self.center_velocity
        part.principal_axes_vectors = self.principal_axes_vectors
        for key in part.keys():
            # part[key].center_position = this_center_position
            # part[key].center_velocity = self.center_velocity
            part[key].principal_axes_vectors = self.principal_axes_vectors

    def _time_in_Myr_(self):
        original_times_in_Gyr = np.array([self.snapshots[i].snapshot['time']
                                          for i in range(len(self.snapshots))])
        time_in_Myr = original_times_in_Gyr * 1000.0
        time_in_Myr -= original_times_in_Gyr[self.initial_key] * 1000.0

        return time_in_Myr

    def _init_starting_star_interpolators_(self):
        self.chosen_indices = []
        self.chosen_snapshot_positions = []
        self.chosen_snapshot_velocities = []

        for i in range(len(self.snapshots)):
            index = int(np.where(self.snapshots[i]['star']['id'] == self.chosen_id)[0])
            self.chosen_indices.append(index)

            position = self.snapshots[i]['star'].prop('host.distance.principal')[index]
            velocity = self.snapshots[i]['star'].prop('host.velocity.principal')[index]
            pos, vel = self._rotate_pos_vel_(position, velocity)
            self.chosen_snapshot_positions.append(pos)
            self.chosen_snapshot_velocities.append(vel)

        self.chosen_indices = np.array(self.chosen_indices)
        self.chosen_snapshot_positions = np.array(self.chosen_snapshot_positions)
        self.chosen_snapshot_velocities = np.array(self.chosen_snapshot_velocities)

        self.chosen_pos_interp = self._gen_pos_or_vel_interpolator_(self.chosen_snapshot_positions)
        self.chosen_vel_interp = self._gen_pos_or_vel_interpolator_(self.chosen_snapshot_velocities)

    def _rotate_pos_vel_(self, p, v):
        theta = np.arctan2(p[0], p[1]) - np.pi/2.0
        ct = np.cos(theta)
        st = np.sin(theta)
        mat = np.array([[ct, -st, 0],[st, ct, 0], [0, 0, 1]])
        return np.matmul(mat, p), np.matmul(mat, v)


    def _gen_pos_or_vel_interpolator_(self, pos_or_vel):
        interpolators = np.zeros(3).tolist()
        for i in range(3):
            interpolators[i] = interpolate.splrep(self.time_in_Myr, pos_or_vel[:,i])
        return interpolators

    def _grid_cache_name_(self, snapshot_index=None):
        if snapshot_index is not None:
            cache_name = 'grid_snapshot'+str(snapshot_index)+'_'
        else:
            cache_name = 'grid_'
        cache_name += self.sim_name
        cache_name += '_ssid' + str(self.chosen_id)
        cache_name += '_gridseed' + str(self.grid_seed) + '_Rmax' + str(self.Rmax)
        cache_name += '_theta' + str(self.theta) + '_grid_x_size' + str(self.grid_x_size_in_kpc)
        cache_name += '_grid_y_size' + str(self.grid_y_size_in_kpc)
        cache_name += '_grid_z_size' + str(self.grid_z_size_in_kpc)
        if self.fine_grid:
            cache_name += '_fine_grid_x_size' + str(self.grid_fine_x_size_in_kpc)
            cache_name += '_fine_grid_y_size' + str(self.grid_fine_y_size_in_kpc)
            cache_name += '_fine_grid_z_size' + str(self.grid_fine_z_size_in_kpc)
            cache_name += '_fine_grid_resolution' + str(self.grid_fine_resolution)
        cache_name += '_start'+str(self.startnum)
        cache_name += '_end'+str(self.endnum)+'_numprior'+str(self.num_prior)
        return cache_name, self.cache_directory + '/' + cache_name

    def _init_grid_(self, grid_snapshot=None):
        grid_cache_name, grid_cache_file = self._grid_cache_name_()
        try:
            self.grid = pickle.load(open(grid_cache_file, 'rb'))
            self._init_acceleration_grid_interpolators_()
            return None
        except:
            print('couldnt find cached grid:')
            print(grid_cache_name)
            print('constructing...')
            pass

        cyl_positions = np.concatenate((self.first_snapshot['star'].prop('host.distance.principal.cylindrical'),
            self.first_snapshot['dark'].prop('host.distance.principal.cylindrical'),
            self.first_snapshot['gas'].prop('host.distance.principal.cylindrical')))

        #self.grid = grid(self.grid_R_min, self.grid_R_max, self.grid_z_max,
        #                 self.grid_phi_size, self.grid_N, self.grid_seed, cyl_positions)
        # old API
        self.grid = grid(self.grid_x_size_in_kpc, self.grid_y_size_in_kpc, self.grid_z_size_in_kpc,
                            self.grid_resolution)

        if self.fine_grid:
            self.grid.add_fine_grid(self.grid_fine_x_size_in_kpc,
                                    self.grid_fine_y_size_in_kpc,
                                    self.grid_fine_z_size_in_kpc,
                                    self.grid_fine_resolution)

        self.grid.snapshot_acceleration_x = []
        self.grid.snapshot_acceleration_y = []
        self.grid.snapshot_acceleration_z = []

        # TODO clean up this section

        # if user specified to calc a snapshot's grid, then do so,
        # dump to cache, and quit
        snap_indices = np.array([self.snapshots[i].snapshot['index']
                                 for i in range(len(self.snapshots))])
        if grid_snapshot is not None:
            snap_cache_name, snap_cache_file = \
                self._grid_cache_name_(grid_snapshot)
            snap_cache_file_x = snap_cache_file.replace('snapshot',
                                                        'snapshot_x')
            snap_cache_file_y = snap_cache_file.replace('snapshot',
                                                        'snapshot_y')
            snap_cache_file_z = snap_cache_file.replace('snapshot',
                                                        'snapshot_z')

            print('generating snapshot grid for this file:')
            print(snap_cache_name)

            key = int(np.where(grid_snapshot == snap_indices)[0])
            position = self.chosen_snapshot_positions[key]
            # velocity = self.chosen_snapshot_velocities[key]
            snap = self.snapshots[key]

            self.grid.gen_evolved_grid(position)

            this_snapshot_grid_x, this_snapshot_grid_y, this_snapshot_grid_z =\
                self._populate_grid_acceleration_(snap, self.grid)

            pickle.dump(this_snapshot_grid_x, open(snap_cache_file_x, 'wb'),
                        protocol=4)
            pickle.dump(this_snapshot_grid_y, open(snap_cache_file_y, 'wb'),
                        protocol=4)
            pickle.dump(this_snapshot_grid_z, open(snap_cache_file_z, 'wb'),
                        protocol=4)
            print('done, will quit now...')
            sys.exit(0)

        for i in range(len(self.snapshots)):
            position = self.chosen_snapshot_positions[i]
            # velocity = self.chosen_snapshot_velocities[i]
            snap = self.snapshots[i]

            self.grid.gen_evolved_grid(position)

            try:
                snap_cache_name, snap_cache_file =\
                    self._grid_cache_name_(snap.snapshot['index'])
                snap_cache_file_x =\
                    snap_cache_file.replace('snapshot', 'snapshot_x')
                snap_cache_file_y =\
                    snap_cache_file.replace('snapshot', 'snapshot_y')
                snap_cache_file_z =\
                    snap_cache_file.replace('snapshot', 'snapshot_z')

                this_snapshot_grid_x =\
                    pickle.load(open(snap_cache_file_x, 'rb'))
                this_snapshot_grid_y =\
                    pickle.load(open(snap_cache_file_y, 'rb'))
                this_snapshot_grid_z =\
                    pickle.load(open(snap_cache_file_z, 'rb'))
            except:
                this_snapshot_grid_x, this_snapshot_grid_y,\
                    this_snapshot_grid_z =\
                    self._populate_grid_acceleration_(snap, self.grid)
                pickle.dump(this_snapshot_grid_x,
                            open(snap_cache_file_x, 'wb'), protocol=4)
                pickle.dump(this_snapshot_grid_y,
                            open(snap_cache_file_y, 'wb'), protocol=4)
                pickle.dump(this_snapshot_grid_z,
                            open(snap_cache_file_z, 'wb'), protocol=4)

            self.grid.snapshot_acceleration_x.append(this_snapshot_grid_x)
            self.grid.snapshot_acceleration_y.append(this_snapshot_grid_y)
            self.grid.snapshot_acceleration_z.append(this_snapshot_grid_z)

        self.grid.snapshot_acceleration_x =\
            np.array(self.grid.snapshot_acceleration_x)
        self.grid.snapshot_acceleration_y =\
            np.array(self.grid.snapshot_acceleration_y)
        self.grid.snapshot_acceleration_z =\
            np.array(self.grid.snapshot_acceleration_z)

        self._init_acceleration_grid_interpolators_()

        pickle.dump(self.grid, open(grid_cache_file, 'wb'), protocol=4)

    def _populate_grid_acceleration_(self, snap, grid):

        # first exclude starting star
        ss_key = np.where(snap['star']['id'] != self.chosen_id)[0]

        # gather all necessary parameters
        all_position = np.concatenate((snap['star'].prop('host.distance.principal')[ss_key],
                snap['dark'].prop('host.distance.principal'),
                snap['gas'].prop('host.distance.principal')))

        # all_velocity = np.concatenate((snap['star'].prop('host.velocity.principal')[ss_key],
        #         snap['dark'].prop('host.velocity.principal'),
        #         snap['gas'].prop('host.velocity.principal')))

        all_mass = np.concatenate((snap['star']['mass'][ss_key],
                snap['dark']['mass'],
                snap['gas']['mass']))

        # set star softening
        if self.star_char_mass is not None:
            star_mass = snap['star']['mass']
            star_softening = np.power(star_mass/self.star_char_mass, 1.0/3.0)
            star_softening /= 1000.0
        else:
            star_softening = np.full(len(snap['star']['position']),
                            float(self.star_softening_in_pc)/1000.0)[ss_key]

        if self.dark_char_mass is not None:
            dark_mass = snap['dark']['mass']
            dark_softening = np.power(dark_mass/self.dark_char_mass, 1.0/3.0)
            dark_softening /= 1000.0
        else:
            dark_softening = np.full(len(snap['dark']['position']),
                float(self.dark_softening_in_pc)/1000.0)

        gas_softening = 2.8 * snap['gas']['smooth.length'] / 1000.0

        all_softening = np.concatenate((star_softening, dark_softening,
                                        gas_softening))

        # figure out which particles to exclude
        # rmag = np.linalg.norm(all_position, axis=1)
        # keys = np.where(rmag < self.Rmax)[0]

        r = all_position
        m = all_mass
        soft = all_softening

        print('constructing tree for gravity calculation')
        tree = ConstructKDTree( np.float64(r), np.float64(m), np.float64(soft))

        print('tree calculated, now evaluating')
        accel = GetAccelParallel(grid.evolved_grid, tree, self.G, self.theta)
        # TODO make this step less hacky
        accel_center = GetAccelParallel(np.array([grid.ss_evolved_position]), tree, self.G, self.theta)[0]

        # remove total acceleration of frame, which we are NOT trying to capture
        accel[:, 0] = np.subtract(accel[:, 0], accel_center[0])
        accel[:, 1] = np.subtract(accel[:, 1], accel_center[1])
        accel[:, 2] = np.subtract(accel[:, 2], accel_center[2])

        return accel[:, 0], accel[:, 1], accel[:, 2]

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
        self._init_acceleration_pool_()

    def _init_acceleration_pool_(self):
        self.acc_pool = Pool(processes=self.ncpu,
                        initializer=init_worker,
                        initargs=(self.grid.grid_accx_interpolators,
                                  self.grid.grid_accy_interpolators,
                                  self.grid.grid_accz_interpolators))

    def _execute_acceleration_grid_interpolators_(self, t):
        evolved_acceleration_x =\
            self.acc_pool.starmap(run_worker_x, [(t, i) for i in
                             range(len(self.grid.grid_accx_interpolators))])
        evolved_acceleration_y =\
            self.acc_pool.starmap(run_worker_y, [(t, i) for i in
                             range(len(self.grid.grid_accy_interpolators))])
        evolved_acceleration_z =\
            self.acc_pool.starmap(run_worker_z, [(t, i) for i in
                             range(len(self.grid.grid_accz_interpolators))])

        self.grid.evolved_acceleration_x = np.array(evolved_acceleration_x)
        self.grid.evolved_acceleration_y = np.array(evolved_acceleration_y)
        self.grid.evolved_acceleration_z = np.array(evolved_acceleration_z)

    def evolve_model(self, time, timestep=None):
        this_t_in_Myr = time.value_in(units.Myr)

        if self.axisymmetric:
            if self.axisymmetric_tevolve:
                self.potential = self._agama_potential_(this_t_in_Myr)
            return None


        self._evolve_starting_star_(this_t_in_Myr)

        position = np.array([0, 0, 0])

        self.grid.gen_evolved_grid(position)
        self._execute_acceleration_grid_interpolators_(this_t_in_Myr)

        print('evolved model to t (Myr):', this_t_in_Myr)

    def evolve_grid(self, pos):
        self.grid.gen_evolved_grid(pos)
        self.grid._grid_evolved_kdtree_ = cKDTree(self.grid.evolved_grid)

    def _evolve_starting_star_(self, time_in_Myr):
        self.chosen_evolved_position = [float(interpolate.splev(time_in_Myr, self.chosen_pos_interp[i])) for i in range(3)]
        self.chosen_evolved_velocity = [float(interpolate.splev(time_in_Myr, self.chosen_vel_interp[i])) for i in range(3)]

        self.chosen_evolved_position = np.array(self.chosen_evolved_position)
        self.chosen_evolved_velocity = np.array(self.chosen_evolved_velocity)

    def _get_pot_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self.grid._grid_evolved_kdtree_.query([x, y, z],
                                                          self.nclose)
        rbfi = RBFInterpolant(self.grid.evolved_grid[ids],
                              self.grid.evolved_potential[ids],
                              basis=self.basis, order=self.order)
        return rbfi

    def _get_acc_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self.grid._grid_evolved_kdtree_.query([x, y, z],
                                                          self.nclose)
        rbfi_x = RBFInterpolant(self.grid.evolved_grid[ids],
                                self.grid.evolved_acceleration_x[ids],
                                basis=self.basis, order=self.order)
        rbfi_y = RBFInterpolant(self.grid.evolved_grid[ids],
                                self.grid.evolved_acceleration_y[ids],
                                basis=self.basis, order=self.order)
        rbfi_z = RBFInterpolant(self.grid.evolved_grid[ids],
                                self.grid.evolved_acceleration_z[ids],
                                basis=self.basis, order=self.order)
        return rbfi_x, rbfi_y, rbfi_z

    def get_gravity_at_point(self, eps, xlist, ylist, zlist):
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if self.axisymmetric:
            pos = np.transpose([xlist, ylist, zlist])
            acc = self.potential.force(pos)
            if len(np.shape(pos))==1:
                ax = acc[0] | (units.kms)**2/units.kpc
                ay = acc[1] | (units.kms)**2/units.kpc
                az = acc[2] | (units.kms)**2/units.kpc
            else:
                ax = acc[:,0] | (units.kms)**2/units.kpc
                ay = acc[:,1] | (units.kms)**2/units.kpc
                az = acc[:,2] | (units.kms)**2/units.kpc
            return ax, ay, az

        if hasattr(xlist, '__iter__'):
            axlist = []
            aylist = []
            azlist = []
            for x, y, z in zip(xlist, ylist, zlist):
                rbfi_x, rbfi_y, rbfi_z = self._get_acc_rbfi_(x, y, z)
                axlist.append(float(rbfi_x([[x, y, z]])))
                aylist.append(float(rbfi_y([[x, y, z]])))
                azlist.append(float(rbfi_z([[x, y, z]])))
            # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
            ax = axlist | units.kms/units.Myr
            ay = aylist | units.kms/units.Myr
            az = azlist | units.kms/units.Myr
            return ax, ay, az

        else:
            rbfi_x, rbfi_y, rbfi_z = self._get_acc_rbfi_(xlist, ylist, zlist)
            # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
            ax = float(rbfi_x([[xlist, ylist, zlist]])) | units.kms/units.Myr
            ay = float(rbfi_y([[xlist, ylist, zlist]])) | units.kms/units.Myr
            az = float(rbfi_z([[xlist, ylist, zlist]])) | units.kms/units.Myr
            return ax, ay, az

    def get_tidal_tensor_at_point(self, eps, xlist, ylist, zlist):
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist, '__iter__'):
            Tlist = []
            for x, y, z in zip(xlist, ylist, zlist):
                rbfi_x, rbfi_y, rbfi_z = self._get_acc_rbfi_(x, y, z)
                Txx = float(rbfi_x([[x, y, z]], diff=(1, 0, 0)))
                Tyy = float(rbfi_y([[x, y, z]], diff=(0, 1, 0)))
                Tzz = float(rbfi_z([[x, y, z]], diff=(0, 0, 1)))
                Txy = float(rbfi_y([[x, y, z]], diff=(1, 0, 0)))
                Tyx = float(rbfi_x([[x, y, z]], diff=(0, 1, 0)))
                Txz = float(rbfi_z([[x, y, z]], diff=(1, 0, 0)))
                Tzx = float(rbfi_x([[x, y, z]], diff=(0, 0, 1)))
                Tyz = float(rbfi_z([[x, y, z]], diff=(0, 1, 0)))
                Tzy = float(rbfi_y([[x, y, z]], diff=(0, 0, 1)))
                T = [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
                Tlist.append(T)
            # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
            return Tlist | units.kms/units.Myr/units.kpc

        else:
            rbfi_x, rbfi_y, rbfi_z = self._get_acc_rbfi_(xlist, ylist, zlist)
            # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
            Txx = float(rbfi_x([[xlist, ylist, zlist]], diff=(1, 0, 0)))
            Tyy = float(rbfi_y([[xlist, ylist, zlist]], diff=(0, 1, 0)))
            Tzz = float(rbfi_z([[xlist, ylist, zlist]], diff=(0, 0, 1)))
            Txy = float(rbfi_y([[xlist, ylist, zlist]], diff=(1, 0, 0)))
            Tyx = float(rbfi_x([[xlist, ylist, zlist]], diff=(0, 1, 0)))
            Txz = float(rbfi_z([[xlist, ylist, zlist]], diff=(1, 0, 0)))
            Tzx = float(rbfi_x([[xlist, ylist, zlist]], diff=(0, 0, 1)))
            Tyz = float(rbfi_z([[xlist, ylist, zlist]], diff=(0, 1, 0)))
            Tzy = float(rbfi_y([[xlist, ylist, zlist]], diff=(0, 0, 1)))
            T = [[Txx, Txy, Txz], [Tyx, Tyy, Tyz], [Tzx, Tzy, Tzz]]
            return T | units.kms/units.Myr/units.kpc

    # TODO clean up starting star
    def _init_starting_star_(self):
        self.chosen_position_z0, self.chosen_index_z0, self.chosen_id = \
                            self.starting_star(self.ss_Rmin, self.ss_Rmax,
                                               self.ss_zmin, self.ss_zmax,
                                               self.ss_agemin_in_Gyr,
                                               self.ss_agemax_in_Gyr,
                                               self.ss_seed)

    def starting_star(self, Rmin, Rmax, zmin, zmax, agemin_in_Gyr,
                      agemax_in_Gyr, seed=1776):
            if self.ss_id is not None:
                pos = self.first_snapshot['star'].prop('host.distance.principal')
                chosen_one = np.where(self.first_snapshot['star']['id'] == \
                                      self.ss_id)[0]
                return pos[chosen_one], chosen_one, self.ss_id

            np.random.seed(seed)

            starages = self.first_snapshot['star'].prop('age')
            pos = self.first_snapshot['star'].prop('host.distance.principal')
            # vel = self.first_snapshot['star'].prop('host.velocity.principal')

            Rstar = np.sqrt(pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1])
            zstar = pos[:, 2]

            agebool = np.logical_and(starages > agemin_in_Gyr,
                                     starages < agemax_in_Gyr)
            Rbool = np.logical_and(Rstar > Rmin, Rstar < Rmax)
            zbool = np.logical_and(zstar > zmin, zstar < zmax)

            totbool = np.logical_and(np.logical_and(agebool, Rbool), zbool)
            keys = np.where(totbool)[0]

            if self.ss_action_cuts:
                np.random.seed(seed)
                starages = self.first_snapshot['star'].prop('age')[keys]
                pos = self.first_snapshot['star'].prop('host.distance.principal')[keys]
                self.first_ag = agama_wrapper(self.options_reader)
                self.first_ag.update_index(self.startnum, snap=self.first_snapshot)
                for ss_id in np.random.permutation(self.first_snapshot['star']['id'][keys]):
                    self.first_ag.update_ss(ss_id)
                    self.chosen_actions = self.first_ag.ss_action()
                    print(self.chosen_actions)
                    Jr = self.chosen_actions[0]
                    Jz = self.chosen_actions[1]
                    # Lz = self.chosen_actions[2]
                    Jrbool = Jr > self.Jr_min and Jr < self.Jr_max
                    Jzbool = Jz > self.Jz_min and Jz < self.Jz_max
                    if Jrbool and Jzbool:
                        pos = self.first_snapshot['star'].\
                                prop('host.distance.principal')
                        chosen_one = np.where(self.first_snapshot['star']['id']
                                              == ss_id)[0]
                        return pos[chosen_one], chosen_one, ss_id

            chosen_one = np.random.choice(keys)
            chosen_id = self.first_snapshot['star']['id'][chosen_one]
            # return pos[chosen_one], vel[chosen_one], chosen_one, chosen_id
            return pos[chosen_one], chosen_one, chosen_id

    def gen_all_agama(self):
        self._gen_axisymmetric_(all_snaps=True)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['acc_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    options_file = sys.argv[1]
    opt = options_reader(options_file)

    if len(sys.argv) == 3:
        snap_index = int(sys.argv[2])
        g = gizmo_interface(opt, snap_index)
    else:
        g = gizmo_interface(opt)
