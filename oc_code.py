import numpy as np
from amuse.units import units, nbody_system
from amuse.ic.kingmodel import new_king_model
from oceanic.options import options_reader
from amuse.lab import new_kroupa_mass_distribution
from scipy import interpolate


class agama_interpolator(object):
    def __init__(self, fnames, times):
        self._fnames_ = fnames
        self._times_ = times
        self._fout_ = '_interpolate_' + str(np.random.randint(2**63)) + self._fnames_[0]

        self._dframes_ = []
        for f in fnames:
            self._dframes_.append(self._read_gen_pot_(f))
        #self._keys_ = self._dframes_[0].keys().tolist().remove('options')
        #self._gen_interpolators_()

    def _read_gen_pot_(self, fname):
        fp = open(fname, 'r')
        fl = fp.readline()
        title = fl.replace('\n', '')
        if title == 'Multipole':
            return self._read_multipole_(fp)
        elif title == 'CylSpline':
            return self._read_cylspline_(fp)

    def _read_multipole_(self, fp):
        options = {}
        data_frame = {}
        for _ in range(3):
            nl = fp.readline().replace('\t#', ' ').replace('\n', '').split(' ')
            options[nl[1]] = nl[0]
        data_frame['options'] = options
        data_frame['type'] = 'Multipole'
        nblocks = 2
        for _ in range(nblocks):
            data = {}
            name = fp.readline()
            header_line = fp.readline()
            print(name)
            print(header_line)
            data['name_line'] = name
            data['header_line'] = header_line
            data_block = []
            for _l in range(int(options['n_radial'])):
                line = fp.readline()
                line_of_floats = [float(l) for l in line.split()]
                data_block.append(line_of_floats)
            data['data_block'] = np.array(data_block)
            data_frame[name.replace('#','').replace('\n','')] = data
            if _ != nblocks-1:
                if fp.readline() != '\n':
                    raise Exception("Agama potential file not in an expected format")
        return data_frame

    def _read_cylspline_(self, fp):
        options = {}
        data_frame = {}
        for _ in range(3):
            nl = fp.readline().replace('\t#', ' ').replace('\n', '').split(' ')
            options[nl[1]] = nl[0]
        data_frame['options'] = options
        data_frame['type'] = 'CylSpline'
        nblocks = 3
        for _ in range(nblocks):
            data = {}
            name = fp.readline()
            mline = fp.readline()
            print(name)
            print(mline)
            data['name_line'] = name
            data['m_line'] = mline
            data_block = []
            for _l in range(int(options['size_R'])+1):
                line = fp.readline()
                line_split = line.split()
                if _l == 0:
                    nan_replacement = line_split[0]
                    line_split[0] = np.nan
                line_of_floats = [float(l) for l in line_split]
                data_block.append(line_of_floats)
            data['data_block'] = np.array(data_block)
            data['nan_replacement'] = nan_replacement
            data_frame[name.replace('#','').replace('\n','')] = data
            if _ != nblocks-1:
                if fp.readline() != '\n':
                    raise Exception("Agama potential file not in an expected format")
        return data_frame

    def _dump_(self, frame, fout):
        if frame['type'] == 'Multipole':
            self._dump_multipole_(frame, fout)
        elif frame['type'] == 'CylSpline':
            self._dump_cylspline_(frame, fout)
        else:
            raise Exception("not recognized type:", frame['type'])

    def _dump_multipole_(self, frame, fout):
        fo = open(fout, 'w')
        fo.write('Multipole\n')

        nradial_line = frame['options']['n_radial'] + '\t#n_radial\n'
        lmax_line = frame['options']['l_max'] + '\t#l_max\n'
        unused_line = frame['options']['unused'] + '\t#unused\n'
        fo.write(nradial_line)
        fo.write(lmax_line)
        fo.write(unused_line)

        name_list = ['Phi', 'dPhi/dr']
        for i,frame_name in enumerate(name_list):
            name_line = frame[frame_name]['name_line']
            header_line = frame[frame_name]['header_line']
            fo.write(name_line)
            fo.write(header_line)
            for line in frame[frame_name]['data_block']:
                dline = ''
                for n in line.tolist():
                    dline += str(n)+'\t'
                dline = dline[:-1] + '\n'
                fo.write(dline)
            if i != len(name_list)-1:
                fo.write('\n')
        fo.close()

    def _dump_cylspline_(self, frame, fout):
        fo = open(fout, 'w')
        fo.write('CylSpline\n')

        sizeR_line = frame['options']['size_R'] + '\t#n_radial\n'
        sizez_line = frame['options']['size_z'] + '\t#l_max\n'
        mmax_line = frame['options']['m_max'] + '\t#unused\n'
        fo.write(sizeR_line)
        fo.write(sizez_line)
        fo.write(mmax_line)

        name_list = ['Phi', 'dPhi/dR', 'dPhi/dz']
        for i,frame_name in enumerate(name_list):
            name_line = frame[frame_name]['name_line']
            m_line = frame[frame_name]['m_line']
            fo.write(name_line)
            fo.write(m_line)
            for j,line in enumerate(frame[frame_name]['data_block']):
                line_list = line.tolist()
                if j==0:
                    if np.isnan(line_list[0]):
                        line_list[0] = frame[frame_name]['nan_replacement']
                    else:
                        raise Exception("Tried replacing nan when dumping, failed")
                dline = ''
                for n in line_list:
                    dline += str(n)+'\t'
                dline = dline[:-1] + '\n'
                fo.write(dline)
            if i != len(name_list)-1:
                fo.write('\n')
        fo.close()

    def _gen_interpolators_(self):
        t = self._times_
        keys = self._dframes_[0].keys()
        interp_frame = {}
        for k in self._keys_:
            all_blocks = np.array([self._dframes_[i][k]['data_block']
                                   for i in range(len(self._dframes_))])
            sp = np.shape(all_blocks)[1:]
            k_int = np.zeros(sp).tolist()
            for i,j in sp:
                k_int[i][j] = interpolate.splrep(t, all_blocks[:, i, j])
            interp_frame[k] = k_int
        self._interp_frame_ = interp_frame

    def _eval_interpolators_(self, t):
        eval_frame = self._dframes_[0].copy()
        for k in self._keys_:
            for i,j in np.shape(eval_frame[k]['data_block']):
                eval_frame[k]['data_block'][i][j] = \
                    interpolate.splev(t, self._interp_frame_[k][i][j])

        self._dump_(eval_frame, self._fout_)

class oc_code(object):
    def __init__(self, options_reader):

        options_reader.set_options(self)
        # self.N, self.W0, self.Mcluster, self.Rcluster, self.ncpu
        # self.gpu_enabled, self.ngpu, self.softening

        if self.use_kroupa is True:
            m = new_kroupa_mass_distribution(self.N,
                                             self.kroupa_max | units.MSun)
            tot_m = np.sum(m.value_in(units.MSun))
            self.converter = \
                nbody_system.nbody_to_si(tot_m | units.MSun,
                                         self.Rcluster | units.parsec)

            self.bodies = new_king_model(self.N, self.W0,
                                         convert_nbody=self.converter)
            self.bodies.mass = m
            self.bodies.scale_to_standard(convert_nbody=self.converter)
            self.bodies.move_to_center()
        else:
            self.converter = \
                nbody_system.nbody_to_si(self.Mcluster | units.MSun,
                                         self.Rcluster | units.parsec)

            self.bodies = new_king_model(self.N, self.W0,
                                         convert_nbody=self.converter)

        if self.gpu_enabled:
            self.code = self.nbodycode(self.converter,
                                       number_of_workers=self.ngpu, mode='gpu')
        else:
            self.code = self.nbodycode(self.converter,
                                       number_of_workers=self.ncpu)

        parameters = {'epsilon_squared': (self.softening | units.parsec)**2}

        for name, value in parameters.items():
            setattr(self.code.parameters, name, value)
        self.code.particles.add_particles(self.bodies)

    def clean_ejections(self, system):
        cluster_positions =\
            np.transpose([system.particles.x.value_in(units.parsec),
                          system.particles.y.value_in(units.parsec),
                          system.particles.z.value_in(units.parsec)])

        median_position = np.median(cluster_positions, axis=0)
        dist = np.linalg.norm(np.subtract(cluster_positions, median_position),
                              axis=1)
        keys = np.where(dist > self.eject_cut)[0]
        if len(keys > 1):
            system.particles.remove_particles(system.particles[keys])
        elif len(keys == 1):
            system.particles.remove_particle(system.particles[keys])
        else:
            return None


if __name__ == '__main__':
    import sys
    options_file = sys.argv[1]
    opt = options_reader(options_file)
    oc = oc_code(opt)

# if __name__ == '__main__':
#     fp = open('potential_id546_m12i_r7100_pot_1', 'r')
#     fl = fp.readline()
#
#     if fl.replace('\n','')=='Multipole':
#         d = _read_multipole_(fp)
#         _dump_multipole_(d, 'test.txt')
#     elif fl.replace('\n','')=='CylSpline':
#         d = _read_cylspline_(fp)
#         l = _dump_cylspline_(d, 'test_1.txt')
