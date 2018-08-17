import numpy as np
from scipy.interpolate import Rbf
from amuse.units import units
from scipy.spatial import cKDTree
from tqdm import tqdm
from scipy.interpolate import griddata

from rbf.interpolate import RBFInterpolant

from joblib import Parallel, delayed
import multiprocessing as mp


def load_pos(interface, index):
    return interface.load_pos(index)

def load_pot(interface, index):
    return interface.load_pot(index)

class cylindrical(object):
    pass

class InterfaceGravity(object):
    def __init__(self, interface, options):
        
        # load in all the positions and potentials
        # pretty mem intensive...
        # not sure if this will work with the parallelization, but will leave for now
        
        self.potentialarray = []
        self.positionarray = []
        for i,index in enumerate(tqdm(interface.index)):
            self.potentialarray.append(interface.load_pot(index))
            self.positionarray.append(interface.load_pos(index))
        
        self.potentialarray = np.array(self.potentialarray)
        self.positionarray = np.array(self.positionarray)

        """
        nproc = 4
        pool = mp.Pool(processes=nproc)
        
        results = [pool.apply(load_pot, args=(interface, index)) for index in tqdm(interface.index)]
        #self.potentialarray = [p.get() for p in results]
        self.potentialarray = results

        results = [pool.apply(load_pos, args=(interface, index)) for index in tqdm(interface.index)]
        #self.positionarray = [p.get() for p in results]
        self.positionarray = results

        #self.potentialarray = Parallel(n_jobs=nproc) (delayed(interface.load_pot)(index) for index in tqdm(interface.index))
        #self.positionarray = Parallel(n_jobs=nproc) (delayed(interface.load_pos)(index) for index in tqdm(interface.index))
        """

        self.original_time_in_Myr = interface.original_time_in_Myr
        self.snapshot_id = interface.index

        self.time_in_Myr = interface.time_in_Myr
        
        keys = self.time_in_Myr.argsort()
        self.positionarray = self.positionarray[keys]
        self.potentialarray = self.potentialarray[keys]
        self.original_time_in_Myr = self.original_time_in_Myr[keys]
        self.time_in_Myr = self.time_in_Myr[keys]
        self.snapshot_id = self.snapshot_id[keys]
        self.nsnapshots = len(self.snapshot_id)

        rcut = options['rcut']
        rcut_keys = self._get_rcut_keys_(rcut, self.positionarray[0])
        self.nparticles = len(rcut_keys)
        for i in range(len(interface.index)):
            self.positionarray[i] = [ self.positionarray[i][ky] for ky in rcut_keys ]
            self.potentialarray[i] = [ self.potentialarray[i][ky] for ky in rcut_keys ]

        self.basis = options['basis']
        self.order = options['order']
        self.nclose = options['nclose']
        #self.evolve_model(0 | units.Myr)

    def _get_rcut_keys_(self, rcut, position):
        #rmagsq = position[:,0]*position[:,0] + position[:,1]*position[:,1] + position[:,2]*position[:,2]
        rmag = np.linalg.norm(position, axis=1)
        rcut_keys = np.where(rmag < rcut)[0]
        return rcut_keys


    def evolve_model(self, time, timestep=None):
        t_in_Myr = time.value_in(units.Myr)
        self.position = []
        self.potential = []
        for i in tqdm(range(self.nparticles)):
            ipartpos = np.array([self.positionarray[j][i] for j in range(self.nsnapshots)])
            xpos = float(griddata(self.time_in_Myr, ipartpos[:,0], t_in_Myr))
            ypos = float(griddata(self.time_in_Myr, ipartpos[:,1], t_in_Myr))
            zpos = float(griddata(self.time_in_Myr, ipartpos[:,2], t_in_Myr))
            self.position.append([xpos,ypos,zpos])

            ipartpot = np.array([self.potentialarray[j][i] for j in range(self.nsnapshots)])
            pot = float(griddata(self.time_in_Myr, ipartpot, t_in_Myr))
            self.potential.append(pot)

        self.position = np.array(self.position)
        self.potential = np.array(self.potential)
        self.kdtree = cKDTree(self.position)
        return None


    def _get_cylindrical_(self):
        out = cylindrical()
        
        xpos = self.position[:,0]
        ypos = self.position[:,1]
        zpos = self.position[:,2]
        xvel = self.velocity[:,0]
        yvel = self.velocity[:,1]
        zvel = self.velocity[:,2]

        out.r = np.sqrt(xpos*xpos + ypos*ypos)
        out.phi = np.mod(-np.arctan2(ypos, xpos), 2*np.pi) # minus sign puts us on cw rotation convention
        out.z = zpos
        out.vr = (xpos*xvel + ypos*yvel)/out.r
        out.vphi = -(xpos*yvel - xvel*ypos)/out.r/out.r # same as above
        out.vz = zvel

        return out

    def _get_rbfi_(self, x, y, z):
        # returns the rbfi interpolator
        # using the user defined number of points, basis, and order
        dist, ids = self.kdtree.query([x,y,z], self.nclose)
        rbfi = RBFInterpolant(self.position[ids], self.potential[ids], basis=self.basis, order=self.order)
        return rbfi

    def get_potential_at_point(self, eps, xlist, ylist, zlist):
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            potlist = []
            for x,y,z in zip(xlist,ylist,zlist):
                rbfi = self._get_rbfi_(x,y,z)
                potlist.append( rbfi( [[x,y,z],[x,y,z]] )[0] )
            return potlist | units.kms * units.kms
        
        else:
            rbfi = self._get_rbfi_(xlist, ylist, zlist)
            return rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]])[0] | units.kms * units.kms
    
    def get_gravity_at_point(self, eps, xlist, ylist, zlist):
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
            ax = axlist | units.kms*units.kms/units.kpc
            ay = aylist | units.kms*units.kms/units.kpc
            az = azlist | units.kms*units.kms/units.kpc
            return ax, ay, az 
        
        else:
            rbfi = self._get_rbfi_(xlist, ylist, zlist)
            ax = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(1,0,0) )[0] | units.kms*units.kms/units.kpc
            ay = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(0,1,0) )[0] | units.kms*units.kms/units.kpc
            az = -rbfi([[xlist, ylist, zlist],[xlist, ylist, zlist]], diff=(0,0,1) )[0] | units.kms*units.kms/units.kpc
            return ax, ay, az

    def get_tidal_at_point(self, eps, xlist, ylist, zlist):
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            Txxlist = []
            Tyylist = []
            Tzzlist = []
            Txylist = []
            Tyzlist = []
            Tzxlist = []
            for x,y,z in zip(xlist,ylist,zlist):
                rbfi = self._get_rbfi_(x,y,z)
                Txxlist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(2,0,0) )[0] )
                Tyylist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(0,2,0) )[0] )
                Tzzlist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(0,0,2) )[0] )
                Txylist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(1,1,0) )[0] )
                Tyzlist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(0,1,1) )[0] )
                Tzxlist.append( -rbfi( [[x,y,z],[x,y,z]], diff=(1,0,1) )[0] )
            Txx = Txxlist
            Tyy = Tyylist
            Tzz = Tzzlist
            Txy = Txylist
            Tyz = Tyzlist
            Tzx = Tzxlist
            return [[Txx, Txy, Tzx],[Txy, Tyy, Tyz],[Tzx, Tyz, Tzz]] | units.kms*units.kms/units.kpc/units.kpc
        
        else:
            rbfi = self._get_rbfi_(xlist, ylist, zlist)
            Txx = -rbfi( [[x,y,z],[x,y,z]], diff=(2,0,0) )[0]
            Tyy = -rbfi( [[x,y,z],[x,y,z]], diff=(0,2,0) )[0]
            Tzz = -rbfi( [[x,y,z],[x,y,z]], diff=(0,0,2) )[0]
            Txy = -rbfi( [[x,y,z],[x,y,z]], diff=(1,1,0) )[0]
            Tyz = -rbfi( [[x,y,z],[x,y,z]], diff=(0,1,1) )[0]
            Tzx = -rbfi( [[x,y,z],[x,y,z]], diff=(1,0,1) )[0]
            return [[Txx, Txy, Tzx],[Txy, Tyy, Tyz],[Tzx, Tyz, Tzz]] | units.kms*units.kms/units.kpc/units.kpc

    def get_tidal_eigen_at_point(self, eps, xlist, ylist, zlist):
        xlist = xlist.value_in(units.kpc)
        ylist = ylist.value_in(units.kpc)
        zlist = zlist.value_in(units.kpc)

        if hasattr(xlist,'__iter__'):
            wlist = []
            vlist = []
            for x,y,z in zip(xlist,ylist,zlist):
                tidal = self.get_tidal_at_point(eps, x, y, z).value_in(1/units.Myr**2)
                w, v = np.linalg.eig(tidal)
                v = v[w.argsort()[::-1]]
                w = w[w.argsort()[::-1]]
                wlist.append(w)
                vlist.append(v)
            return (wlist | 1/units.Myr**2, vlist)       
        else:
            tidal = self.get_tidal_at_point(eps, xlist, ylist, zlist).value_in(1/units.Myr**2)
            w, v = np.linalg.eig(tidal)
            v = v[w.argsort()[::-1]]
            w = w[w.argsort()[::-1]]
            return (w | 1/units.Myr**2, v)

    def starting_star(self, Rmin, Rmax, zmin, zmax, agemax_in_Gyr, seed):
            return self.interface.starting_star(Rmin, Rmax, zmin, zmax, agemax_in_Gyr, seed)
    
if __name__ == '__main__':
    from rbf.basis import phs3
    import NIHAO_interface

    options = {'rcut': 30,
                'basis': phs3,
                'order': 5,
                'nclose': 250
                }

    #nihao = NIHAO_interface.NIHAO_interface('/Volumes/abeane_drive2/tobias/tobias', 1800, 1820)
    nihao = NIHAO_interface.NIHAO_interface('/mnt/ceph/users/abeane/download/NIHAO_2.79e12', 1800, 1803)
    g = InterfaceGravity(nihao, options)
    #g = LatteGravity('position.npy', 'potential.npy', 'velocity.npy', phs3, 5, 0.4 | units.kpc)

