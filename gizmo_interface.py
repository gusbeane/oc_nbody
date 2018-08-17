import numpy as np
import gizmo_analysis as gizmo
import utilities as ut
from scipy import interpolate
from scipy.spatial import cKDTree
from rbf.interpolate import RBFInterpolant
from rbf.basis import phs3
#from colossus.cosmology import cosmology

class gizmo_interface(object):
    def __init__(self, directory, startnum, endnum,
                    one_prior=True):
        
        self.convert_kms_Myr_to_kpc = 20000.0*np.pi / (61478577.0) # thanks wolfram alpha

        # TODO make these user-definable
        self.nclose = 100
        self.basis = phs3
        self.order = 5

        self.directory = directory
        self.startnum = startnum
        self.endnum = endnum
        
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

        # recenter first snapshot
        self._recenter_snapshot_(self.first_snapshot)

        # read in all snapshots, but only the necessary quantities, and recenter
        if one_prior:
            self.snapshot_indices = range(startnum-1, endnum+1)
            self.initial_key = 1
        else:
            self.snapshot_indices = range(startnum, endnum+1)
            self.initial_key = 0


        self.snapshots = gizmo.io.Read.read_snapshots(['star','dark'], 'index', self.snapshot_indices, 
                                                        properties=['position', 'potential'], 
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
        
        # generate potential, position arrays
        self.position_array = self._position_array_()
        self.potential_array = self._potential_array_()

        # set up interpolator
        self.position_interpolator = self._init_position_interpolators_()
        self.potential_interpolator = self._init_potential_interpolators_()

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


    def _init_position_interpolators_(self):
        interpolators = np.zeros(np.shape(self.position_array)[0:2]).tolist()
        for i in range(len(interpolators)):
            for j in range(3):
                interpolators[i][j] = interpolate.splrep(self.time_in_Myr, self.position_array[i][j])
        return interpolators
    
    def _init_potential_interpolators_(self):
        interpolators = np.zeros(np.shape(self.potential_array)[0:2]).tolist()
        for i in range(len(interpolators)):
            interpolators[i] = interpolate.splrep(self.time_in_Myr, self.potential_array[i])
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

    def evolve_model(self, time, timestep=None):
        # TODO parallelize this function

        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        # t_in_Myr = time.value_in(units.Myr)
        t_in_Myr = time
        self.evolved_position = np.zeros(np.shape(self.position_array)[0:2])
        self.evolved_potential = np.zeros(np.shape(self.potential_array)[0:1])
        for i in range(len(self.evolved_position)):
            for j in range(3):
                self.evolved_position[i][j] = interpolate.splev(t_in_Myr, self.position_interpolator[i][j])
            self.evolved_potential[i] = interpolate.splev(t_in_Myr, self.potential_interpolator[i])
        
        rmag = np.linalg.norm(self.evolved_position, axis=1)
        keys = np.where(np.logical_and(rmag > 3, rmag < 13))[0]
        self.evolved_position = self.evolved_position[keys]
        self.evolved_potential = self.evolved_potential[keys]
        self._kdtree_ = cKDTree(self.evolved_position)

    def _get_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self._kdtree_.query([x,y,z], self.nclose)
        rbfi = RBFInterpolant(self.evolved_position[ids], self.evolved_potential[ids], basis=self.basis, order=self.order)
        return rbfi
    
    def get_potential_at_point(self, eps, xlist, ylist, zlist):
        # UNCOMMENT THIS WHEN IMPLEMENT AMUSE
        # xlist = xlist.value_in(units.kpc)
        # ylist = ylist.value_in(units.kpc)
        # zlist = zlist.value_in(units.kpc)

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
        #xlist = xlist.value_in(units.kpc)
        #ylist = ylist.value_in(units.kpc)
        #zlist = zlist.value_in(units.kpc)

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


    def starting_star(self, Rmin, Rmax, zmin, zmax, agemax_in_Gyr=1, seed=1776):
        np.random.seed(seed)
        starages = self.first_snapshot['star'].prop('age')
        pos = self.first_snapshot['star']['position']
        vel = self.first_snapshot['star']['velocity']

        Rstar = np.sqrt(pos[:,0]*pos[:,0] + pos[:,1]*pos[:,1])
        zstar = pos[:,2]

        agebool = starages < agemax_in_Gyr
        Rbool = np.logical_and(Rstar > Rmin, Rstar < Rmax)
        zbool = np.logical_and(zstar > zmin, zstar < zmax)

        totbool = np.logical_and(np.logical_and(agebool,Rbool), zbool)
        keys = np.where(totbool)[0]

        chosen_one = np.random.choice(keys)
        return pos[chosen_one], vel[chosen_one], chosen_one


if __name__ == '__main__':
    import sys
    simulation_directory = sys.argv[1]
    startnum = int(sys.argv[2])
    endnum = int(sys.argv[3])
    g = gizmo_interface(simulation_directory, startnum, endnum)
