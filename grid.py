import numpy as np
from tqdm import tqdm

"""
module for grid calculations

should not need to be imported by the user

"""

class grid(object):
    def __init__(self, x_size_in_kpc, y_size_in_kpc, z_size_in_kpc, resolution, Rmag):
        self.x_size_in_kpc = x_size_in_kpc
        self.y_size_in_kpc = y_size_in_kpc
        self.z_size_in_kpc = z_size_in_kpc
        self.resolution = resolution
        self.Rmag = Rmag
        self.x_n = int(x_size_in_kpc/resolution)
        self.y_n = int(y_size_in_kpc/resolution)
        self.z_n = int(z_size_in_kpc/resolution)
        
        self.x_grid = np.linspace(-x_size_in_kpc, x_size_in_kpc, num=self.x_n)
        self.y_grid = np.linspace(-y_size_in_kpc, y_size_in_kpc, num=self.y_n)
        self.z_grid = np.linspace(-z_size_in_kpc, z_size_in_kpc, num=self.z_n)
        self._gen_init_grid_()

    def update_evolved_grid(self, position, velocity):
        self.evolved_position = position
        self.evolved_velocity = velocity
        xpos = position[0]
        ypos = position[1]
        zpos = position[2]
        """
        ctheta = xvel/np.sqrt(xvel**2. + zvel**2.)
        stheta = zvel/np.sqrt(xvel**2. + zvel**2.)
        first_matrix = np.array([[ctheta, 0.0, -stheta],
                                 [  0.0, 1.0, 0.0],
                                  [ stheta, 0.0, ctheta]])
        """

        vec_mag = np.sqrt(xpos**2. + ypos**2.)
        # ctheta = xpos/vec_mag
        # stheta = ypos/vec_mag
        ctheta = ypos/vec_mag
        stheta = xpos/vec_mag

        second_matrix = np.array([[ctheta, -stheta, 0.0],
                                  [stheta, ctheta, 0.0],
                                   [0.0, 0.0, 1.0]])

        # self._matrix_transform_ = np.matmul(second_matrix, first_matrix)
        self._matrix_transform_ = second_matrix

        self._offset_ = self.Rmag*np.array([xpos, ypos, 0.0])/vec_mag

        self.evolved_grid = np.transpose(np.tensordot(self._matrix_transform_, np.transpose(self.init_grid), axes=1))
        position[2] = 0
        self.evolved_grid = np.add(self.evolved_grid, self._offset_)

    def _gen_init_grid_(self):
        grid_positions = []
        for i in tqdm(range(self.x_n)):
            for j in range(self.y_n):
                for k in range(self.z_n):
                    grid_positions.append([self.x_grid[i], self.y_grid[j], self.z_grid[k]])
        self.init_grid = np.ascontiguousarray(grid_positions)

    def grid_point(self, i, j, k):
        xinit = self.x_grid[i]
        yinit = self.y_grid[j]
        zinit = self.z_grid[k]
        center_pos = np.array([xinit, yinit, zinit])
        center_pos = np.matmul(self._matrix_transform_, center_pos)

        return center_pos + self._offset_


if __name__ == '__main__':
    g = grid(0.3, 3.0, 2.0, 0.03, 8.0)
