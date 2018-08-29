from configparser import ConfigParser

class options_reader(object):
    def __init__(self, file):
        self.parser = ConfigParser(inline_comment_prefixes=';')
        try:
            self.parser.read(file)
        except:
            print('couldnt read options file')
            raise Exception('Cant read the provided options file:' + file)

        self.options = {}
        
        # read in simulation parameters
        for opt in ['simulation_directory', 'cache_directory', 'startnum', 'endnum']:
            self._read_required_option_('simulation', opt)

        self._read_optional_option_('simulation', 'num_prior', '3')

        # read in interpolation parameters
        self._read_optional_option_('interpolation', 'nclose', '150')
        self._read_optional_option_('interpolation', 'basis', 'phs3')
        self._read_optional_option_('interpolation', 'order', '5')

        # read in starting_star parameters
        for opt in ['ss_Rmin', 'ss_Rmax', 'ss_zmin', 'ss_zmax']:
            self._read_required_option_('starting_star', opt)

        self._read_optional_option_('starting_star', 'ss_agemin_in_Gyr', '0')
        self._read_optional_option_('starting_star', 'ss_agemax_in_Gyr', '0')
        self._read_optional_option_('starting_star', 'ss_seed', '1776')

        # read in grid parameters
        for opt in ['grid_x_size', 'grid_y_size', 'grid_z_size', 'grid_resolution']:
            self._read_required_option_('grid', opt)

        self._read_optional_option_('grid', 'grid_buffer', '25.0')

        # convert relevant parameters
        int_options = ['startnum', 'endnum', 'num_prior', 'nclose', 'order', 'ss_seed']
        for opt in int_options:
            if opt in self.options.keys():
                self.options[opt] = int(self.options[opt])

        float_options = ['ss_Rmin', 'ss_Rmax', 'ss_zmin', 'ss_zmax',
                         'grid_x_size', 'grid_y_size', 'grid_z_size', 'grid_buffer', 
                         'grid_resolution',
                         'ss_agemin_in_Gyr', 'ss_agemax_in_Gyr']
        for opt in float_options:
            if opt in self.options.keys():
                self.options[opt] = float(self.options[opt])

        self._convert_rbf_basis_(self.options['basis'])

        for s in ['x', 'y', 'z']:
            val = self.options['grid_'+s+'_size']
            val += self.options['grid_buffer']
            self.options['grid_'+s+'_max'] = val

    def set_options(self, object):
        for key in self.options.keys():
            setattr(object, key, self.options[key])

    def _read_required_option_(self, category, option):
        try:
            self.options[option] = self.parser.get(category, option)
            print('set option: ', option, ' as: ', self.options[option])
        except:
            raise Exception('Couldnt find required option: ' + option)

    def _read_optional_option_(self, category, option, default):
        try:
            self.options[option] = self.parser.get(category, option)
            print('set option: ', option, ' as: ', self.options[option])
        except:
            print('Couldnt find option', option, ', using default: ', default)
            self.options[option] = default

    def _convert_rbf_basis_(self, basis_string):
        if basis_string == 'phs8':
            from rbf.basis import phs8
            self.options['basis'] = phs8
        elif basis_string =='phs7':
            from rbf.basis import phs7
            self.options['basis'] = phs7
        elif basis_string =='phs6':
            from rbf.basis import phs6
            self.options['basis'] = phs6
        elif basis_string =='phs5':
            from rbf.basis import phs5
            self.options['basis'] = phs5
        elif basis_string =='phs4':
            from rbf.basis import phs4
            self.options['basis'] = phs4
        elif basis_string =='phs3':
            from rbf.basis import phs3
            self.options['basis'] = phs3
        elif basis_string =='phs2':
            from rbf.basis import phs2
            self.options['basis'] = phs2
        elif basis_string =='phs1':
            from rbf.basis import phs1
            self.options['basis'] = phs1
        elif basis_string == 'mq':
            from rbf.basis import mq
            self.options['basis'] = mq
        elif basis_string =='imq':
            from rbf.basis import imq
            self.options['basis'] = imq
        elif basis_string =='iq':
            from rbf.basis import iq
            self.options['basis'] = iq
        elif basis_string =='ga':
            from rbf.basis import ga
            self.options['basis'] = ga
        elif basis_string =='exp':
            from rbf.basis import exp
            self.options['basis'] = exp
        elif basis_string =='se':
            from rbf.basis import se
            self.options['basis'] = se
        elif basis_string =='mat32':
            from rbf.basis import mat32
            self.options['basis'] = mat32
        elif basis_string =='mat52':
            from rbf.basis import mat52
            self.options['basis'] = mat52
        elif basis_string =='wen10':
            from rbf.basis import wen10
            self.options['basis'] = wen10
        elif basis_string =='wen11':
            from rbf.basis import wen11
            self.options['basis'] = wen11
        elif basis_string =='wen12':
            from rbf.basis import wen12
            self.options['basis'] = wen12
        elif basis_string =='wen30':
            from rbf.basis import wen30
            self.options['basis'] = wen30
        elif basis_string =='wen31':
            from rbf.basis import wen31
            self.options['basis'] = wen31
        elif basis_string =='wen32':
            from rbf.basis import wen32
            self.options['basis'] = wen32
        else:
            raise Exception("Can't recognize given basis: "+basis_string)

if __name__ == '__main__':
    import sys
    g = options_reader(sys.argv[1])