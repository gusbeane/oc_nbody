import numpy as np

"""
module for grid calculations

should not need to be imported by the user

TODO:
 * change to be spherical grid
    * implement adaptive resolution (not sure best way)

"""


class grid(object):
    def __init__(self, x_size_in_kpc, y_size_in_kpc, z_size_in_kpc,
                 resolution):
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

    def add_fine_grid(self, x_size_in_kpc, y_size_in_kpc, z_size_in_kpc,
                      resolution):
        self.fine_x_size_in_kpc = x_size_in_kpc
        self.fine_y_size_in_kpc = y_size_in_kpc
        self.fine_z_size_in_kpc = z_size_in_kpc
        self.fine_resolution = resolution

        self.x_n = int(x_size_in_kpc/resolution)
        self.y_n = int(y_size_in_kpc/resolution)
        self.z_n = int(z_size_in_kpc/resolution)

        self.x_fine_grid = np.linspace(-x_size_in_kpc, x_size_in_kpc,
                                       num=self.x_n)
        self.y_fine_grid = np.linspace(-y_size_in_kpc, y_size_in_kpc,
                                       num=self.y_n)
        self.z_fine_grid = np.linspace(-z_size_in_kpc, z_size_in_kpc,
                                       num=self.z_n)

        self._remove_coarse_points_()
        self._add_fine_grid_()

    def gen_evolved_grid(self, position):
        self.ss_evolved_position = position
        self.evolved_grid = np.add(self.init_grid, position)

    def _gen_init_grid_(self):
        grid_positions = []
        for i in range(self.x_n):
            for j in range(self.y_n):
                for k in range(self.z_n):
                    grid_positions.append([self.x_grid[i],
                                           self.y_grid[j], self.z_grid[k]])
        # origin keeps total acc on cluster zero
        grid_positions.append([0, 0, 0])

        self.init_grid = np.ascontiguousarray(grid_positions)

    def _remove_coarse_points_(self):
        '''
        removes coarse points to be replaced by the finer grid
        '''
        xbool = np.abs(self.init_grid[:, 0]) < self.fine_x_size_in_kpc
        ybool = np.abs(self.init_grid[:, 1]) < self.fine_y_size_in_kpc
        zbool = np.abs(self.init_grid[:, 2]) < self.fine_z_size_in_kpc
        in_bool = np.logical_and(np.logical_and(xbool, ybool), zbool)
        keys = np.where(np.logical_not(in_bool))[0]
        self.init_grid = self.init_grid[keys]

    def _add_fine_grid_(self):
        grid_positions = self.init_grid.tolist()
        for i in range(self.x_n):
            for j in range(self.y_n):
                for k in range(self.z_n):
                    grid_positions.append([self.x_fine_grid[i],
                                           self.y_fine_grid[j],
                                           self.z_fine_grid[k]])
        grid_positions.append([0, 0, 0])
        self.init_grid = np.ascontiguousarray(grid_positions)


if __name__ == '__main__':
    g = grid(1, 1, 1, 0.005)
