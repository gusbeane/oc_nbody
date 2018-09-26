import numpy as np

"""
module for grid calculations

should not need to be imported by the user

TODO:
 * change to be spherical grid
    * implement adaptive resolution (not sure best way)

"""

class grid(object):
    def __init__(self, x_size_in_kpc, y_size_in_kpc, z_size_in_kpc, resolution):
        # store parameters for future convenience
        self.x_size_in_kpc = x_size_in_kpc
        self.y_size_in_kpc = y_size_in_kpc
        self.z_size_in_kpc = z_size_in_kpc
        self.resolution = resolution

        # convert resolution to number of grid points needed
        self.x_n = int(x_size_in_kpc/resolution)
        self.y_n = int(y_size_in_kpc/resolution)
        self.z_n = int(z_size_in_kpc/resolution)

        self.x_grid = np.linspace(-x_size_in_kpc, x_size_in_kpc, num=self.x_n)
        self.y_grid = np.linspace(-y_size_in_kpc, y_size_in_kpc, num=self.y_n)
        self.z_grid = np.linspace(-z_size_in_kpc, z_size_in_kpc, num=self.z_n)
        self._gen_init_grid_()

    def gen_evolved_grid(self, position):
        self.ss_evolved_position = position
        self.evolved_grid = np.add(self.init_grid, position)

    def _gen_init_grid_(self):
        grid_positions = []
        for i in range(self.x_n):
            for j in range(self.y_n):
                for k in range(self.z_n):
                    grid_positions.append([self.x_grid[i], self.y_grid[j], self.z_grid[k]])
        # origin keeps total acc on cluster zero
        grid_positions.append([0, 0, 0])

        self.init_grid = np.ascontiguousarray(grid_positions)

if __name__ == '__main__':
    g = grid(0.3, 3.0, 2.0, 0.03, 8.0)
