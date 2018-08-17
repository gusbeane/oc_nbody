import numpy as np
from colossus.cosmology import cosmology
import sys

snapshot_id = range(1800,1820)
index = range(len(snapshot_id))

params = {'flat': True, 'H0': 67.1, 'Om0': 0.3175, 'Ob0': 0.049, 'Ode0': 0.6824, 'sigma8': 0.8344, 'ns': 0.9624}
cosmology.addCosmology('NIHAO', params)
cosmo = cosmology.setCosmology('NIHAO')

posfilearray = []
velfilearray = []
potfilearray = []
original_time_in_Myr = []
for i,snapid in enumerate(snapshot_id):
	# generate file names
	posfname = '2.79e12.' + '0' + str(snapid) + '_position.npy'
	velfname = '2.79e12.' + '0' + str(snapid) + '_velocity.npy'
	potfname = '2.79e12.' + '0' + str(snapid) + '_potential.npy'
	posfilearray.append(posfname)
	velfilearray.append(velfname)
	potfilearray.append(potfname)

	# store time
	datfname = '2.79e12.' + '0' + str(snapid) + '_halo_data.npy'
	dat = np.load(datfname, encoding='latin1')
	redshift = dat[4]
	time = cosmo.age(redshift) * 1000.0 # convert from Gyr to Myr
	original_time_in_Myr.append(time)
	
	if snapid==1800:
		earliest_datfname = datfname

firstsnap = sys.argv[1]

outarray = [posfilearray, velfilearray, potfilearray, list(index), list(snapshot_id), original_time_in_Myr, earliest_datfname, firstsnap]
np.save('2.79e12_sim_info', outarray)