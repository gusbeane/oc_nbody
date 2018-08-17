import numpy as np
import pynbody
from colossus.cosmology import cosmology

class NIHAO_interface(object):
    def __init__(self, directory, startnum, endnum):
        self.directory = directory
        
        # read in pos, vel of first snapshot
        halodat = np.load(self.directory + '/2.79e12.' + "{0:0=5d}".format(startnum) + '_halo_data.npy', encoding='latin1')

        # read in first snapshot, get rotation matrix
        snapname = self.directory+'/2.79e12.'+ "{0:0=5d}".format(startnum)
        snap = pynbody.load(snapname)
        snap.physical_units()

        halos = snap.halos()
        h1 = halos[1]

        self.compos = pynbody.analysis.halo.center(h1, mode='hyb', retcen=True)
        snap['pos'] -= self.compos
        self.comvel = pynbody.analysis.halo.vel_center(h1, mode='hyb', retcen=True)
        snap['vel'] -=self.comvel
        
        self.compos = np.array(self.compos.in_units('kpc').tolist())
        self.comvel = np.array(self.comvel.in_units('km s**-1').tolist())

        transformation = pynbody.analysis.angmom.faceon(h1)
        self.rot_matrix = transformation.matrix
        self.snap = snap
        self.h1 = h1

        self._transformation_ = transformation

        params = {'flat': True, 'H0': 67.1, 'Om0': 0.3175, 'Ob0': 0.049, 'Ode0': 0.6824, 'sigma8': 0.8344, 'ns': 0.9624}
        cosmology.addCosmology('NIHAO', params)
        self.cosmo = cosmology.setCosmology('NIHAO')

        self.index = np.array((range(startnum, endnum + 1)))
        self.redshifts = np.zeros(len(self.index))
        self.original_time_in_Myr = np.zeros(len(self.index))
        self.halodat = []
        for i,thisid in enumerate(self.index):
            halodat = np.load(self.directory + '/2.79e12.' + "{0:0=5d}".format(thisid) + '_halo_data.npy')
            z = halodat[4]
            self.redshifts[i] = z
            self.original_time_in_Myr[i] = self.cosmo.age(z) * 1000.0
            self.halodat.append(halodat)

        self.original_time_in_Myr = np.array(self.original_time_in_Myr)
        self.time_in_Myr = self.original_time_in_Myr - np.min(self.original_time_in_Myr)

    def load_pos(self, index):
        mykey = np.where(self.index == index)[0]
        halo_dat_name = self.directory + '/2.79e12.'+"{0:0=5d}".format(index)+'_halo_data.npy'
        fname = self.directory + '/2.79e12.'+"{0:0=5d}".format(index)+'_position.npy'
        posdat = np.load(fname).tolist()

        # 978 factor converts km/s * Myr to kpc
        offset = (self.compos + self.comvel * self.time_in_Myr[mykey] / 978.461942)
        posdat = [posdat[i] - offset for i in range(len(posdat))] 
        posdat = [np.matmul(self.rot_matrix,posdat[i]).tolist() for i in range(len(posdat))]
        
        return posdat

    def load_pot(self, index):
        mykey = np.where(self.index == index)[0]
        halo_dat_name = self.directory + '/2.79e12.'+"{0:0=5d}".format(index)+'_halo_data.npy'
        fname = self.directory + '/2.79e12.'+"{0:0=5d}".format(index)+'_potential.npy'
        potdat = np.load(fname).tolist()
        return potdat

    def starting_star(self, Rmin, Rmax, zmin, zmax, agemax_in_Gyr, seed):
        np.random.seed(seed)
        starages = self.snap.star['age'].in_units('Gyr').tolist()
        pos = self.snap.star['pos'].in_units('kpc').tolist()
        vel = self.snap.star['vel'].in_units('km s**-1').tolist()
        pos = np.array(pos)
        vel = np.array(vel)

        Rstar = np.sqrt(pos[:,0]*pos[:,0] + pos[:,1]*pos[:,1])
        zstar = pos[:,2]

        agebool = np.array(starages) < agemax_in_Gyr
        Rbool = np.logical_and(Rstar > Rmin, Rstar < Rmax)
        zbool = np.logical_and(zstar > zmin, zstar < zmax)

        totbool = np.logical_and(np.logical_and(agebool,Rbool), zbool)
        keys = np.where(totbool)[0]

        chosen_one = np.random.choice(keys)
        return pos[chosen_one], vel[chosen_one], chosen_one


if __name__ == '__main__':
    import sys
    g = NIHAO_interface(sys.argv[1], 1800, 1820)
