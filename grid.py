import numpy as np
from tqdm import tqdm

"""
module for grid calculations

should not need to be imported by the user

"""

class grid(object):
	def __init__(self, size_in_kpc, resolution, resolution_in_kpc=False):
		self.size_in_kpc = size_in_kpc
		self.resolution = resolution
		if resolution_in_kpc:
			self.resolution_unit = 'kpc'
			self.n = int(size_in_kpc/resolution)
		else:
			self.resolution_unit = 'pc'
			self.n = int(1000.0*size_in_kpc/resolution)
		
		self.oned_grid = np.linspace(-size_in_kpc, size_in_kpc, num=self.n)
		self._gen_init_grid_()

	def update_evolved_grid(self, position, velocity):
		self.evolved_position = position
		self.evolved_velocity = velocity
		xvel = velocity[0]
		yvel = velocity[1]
		zvel = velocity[2]
		ctheta = xvel/np.sqrt(xvel**2. + zvel**2.)
		stheta = zvel/np.sqrt(xvel**2. + zvel**2.)
		first_matrix = np.array([[ctheta, 0.0, -stheta],
								 [  0.0, 1.0, 0.0],
								  [ stheta, 0.0, ctheta]])

		ctheta = xvel/np.sqrt(xvel**2. + yvel**2.)
		stheta = yvel/np.sqrt(xvel**2. + yvel**2.)

		second_matrix = np.array([[ctheta, -stheta, 0.0],
								  [stheta, ctheta, 0.0],
								   [0.0, 0.0, 1.0]])

		self._matrix_transform_ = np.matmul(second_matrix, first_matrix)

		self.evolved_grid = np.transpose(np.tensordot(self._matrix_transform_, np.transpose(self.init_grid), axes=1))
		self.evolved_grid = np.add(self.evolved_grid, position)

	def _gen_init_grid_(self):
		grid_positions = []
		for i in tqdm(range(self.n)):
			for j in range(self.n):
				for k in range(self.n):
					grid_positions.append([self.oned_grid[i], self.oned_grid[j], self.oned_grid[k]])
		self.init_grid = np.ascontiguousarray(grid_positions)

	def grid_point(self, i, j, k):
		xinit = self.oned_grid[i]
		yinit = self.oned_grid[j]
		zinit = self.oned_grid[k]
		center_pos = np.array([xinit, yinit, zinit])
		center_pos = np.matmul(self._matrix_transform_, center_pos)

		return center_pos + position


if __name__ == '__main__':
	g = grid(0.5, 25)