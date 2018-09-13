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

        cut_cyl_positions = self._make_cuts_(cyl_positions)

        self._gen_init_grid_(cut_cyl_positions)

    def _make_cuts_(self, cyl_positions):
        rbool = np.logical_and(cyl_positions[:,0] > self.Rmin, cyl_positions[:,0] < self.Rmax)
        zbool = np.abs(cyl_positions[:,1]) < self.zcut
        keys = np.where(np.logical_and(rbool,zbool))[0]
        return cyl_positions[keys]

    def update_evolved_grid(self, phi):
        ctheta = np.cos(phi)
        stheta = np.sin(phi)

        self._matrix_transform_ = np.array([[ctheta, -stheta, 0.0],
                                            [stheta, ctheta, 0.0],
                                            [0.0, 0.0, 1.0]]) 

        self.evolved_grid = np.transpose(np.tensordot(self._matrix_transform_, np.transpose(self.init_grid), axes=1))

    def _gen_init_grid_(self, cut_cyl_positions):
        nmem = len(cut_cyl_positions)
        grid_keys = np.random.choice(range(nmem), size=self.N, replace=False)
        init_grid = cut_cyl_positions[grid_keys]

        R = init_grid[:,0]
        z = init_grid[:,1]
        phi = (np.random.rand(self.N) * self.phicut) - self.phicut/2.0

        x = R*np.cos(phi)
        y = R*np.sin(phi)

        init_grid = np.transpose([x,y,z])

        self.init_grid = np.ascontiguousarray(init_grid)

    def grid_point(self, i, j, k):
        xinit = self.x_grid[i]
        yinit = self.y_grid[j]
        zinit = self.z_grid[k]
        center_pos = np.array([xinit, yinit, zinit])
        center_pos = np.matmul(self._matrix_transform_, center_pos)

        return center_pos + self._offset_


if __name__ == '__main__':
    g = grid(0.3, 3.0, 2.0, 0.03, 8.0)
