import numpy as np

"""
module for grid calculations

should not need to be imported by the user

"""

class grid(object):
    def __init__(self, Rmin, Rmax, zcut, phicut, N, cyl_positions):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.zcut = zcut
        self.phicut = phicut
        self.N = int(N)

        cut_positions = self._make_cuts_(cyl_positions)

        self._gen_init_grid_(cut_positions)

    def _make_cuts_(self, cyl_positions):
        rbool = np.logical_and(cyl_positions[:,0] > Rmin, cyl_positions[:,0] < Rmax)
        zbool = np.abs(cyl_positions[:,1]) < zcut
        keys = np.where(np.logical_and(rbool,zbool))[0]
        return cyl_positions[keys]

    def update_evolved_grid(self, position, velocity):
        self.evolved_position = position
        self.evolved_velocity = velocity
        
        xpos = position[0]
        ypos = position[1]

        vec_mag = np.sqrt(xpos**2. + ypos**2.)
        ctheta = xpos/vec_mag
        stheta = ypos/vec_mag

        self._matrix_transform_ = np.array([[ctheta, -stheta, 0.0],
                                            [stheta, ctheta, 0.0],
                                            [0.0, 0.0, 1.0]])

        self.evolved_grid = self.init_grid + np.array([0, vec_mag, 0])
        #self.evolved_grid[:,0] += vec_mag
        self.evolved_grid = np.transpose(np.tensordot(self._matrix_transform_, np.transpose(self.evolved_grid), axes=1))

    def _gen_init_grid_(self):
        grid_positions = []
        for i in range(self.x_n):
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
