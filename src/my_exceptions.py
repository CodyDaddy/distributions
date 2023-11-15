class WrongValue(Exception):
    def __init__(self, expected, received=None, name=''):
        super().__init__('[ERROR] Wrong value {} \n expected {}, but received {}'.format(name, expected, received))


class WrongMode(Exception):
    def __init__(self):
        super().__init__('[ERROR] Wrong mode detected!\n '
                         'allowed modes: 0=density distribution, 1=cumulative distribution')


class MissingValues(Exception):
    def __init__(self, x, y, printValues=False):
        if printValues:
            optionalText = '{} \n {}'.format(x, y)
        else:
            optionalText = ''
            super().__init__('[ERROR] missing values detected!\n '
                             'xi and yi need to have the same size\n'
                             'but received: x = {} and y = {} {}\n'.format(len(x), len(y), optionalText))


class IncompatibleDistributions(Exception):
    def __init__(self, d1, d2):
        super().__init__('[ERROR] Distributions incompatible! \n'
                         'Check base, mode and x_type'
                         'd1 = {} \n d2 = {}'.format(d1, d2))


class IncompatibleGrid(Exception):
    def __init__(self, g1, g2):
        super().__init__('\n ---------------------------- \n'
                         '[ERROR] Grids incompatible! \n'
                         'Object 1: {}, grid_V: {} \n'
                         'Object 2: {}, grid_V: {} \n'
                         'Difference: {}'.format(g1.__str__(), g1.x, g2.__str__(), g2.x, g1.x==g2.x))
