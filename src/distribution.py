from collections.abc import Iterable

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import copy
import scipy.interpolate as interpol


# import custom packages
from src import io_node, algorithms as alg
from src import my_exceptions as exc
from scipy.stats import lognorm

modes = [0, 1]  # 0=density distribution, 1=cumulative distribution
x_types = ['#', 'x', 'A', 'V']  # number, length, surface, volume
DistributionNames = ['q', 'Q']
x_names = ['', '_max']


def get_x_i(x_max: numpy.ndarray) -> numpy.array:
    """
    Calculates cell average grid values from given grid

    Parameters
    ----------
    x_max : numpy.ndarray
        Array of grid borders

    Returns
    -------
    xi : numpy.ndarray
        Cell average of grid
    """
    return np.append(x_max[0] / 2, (x_max[:-1] + x_max[1:]) / 2)


def get_dx_i(x_max: numpy.ndarray) -> numpy.array:
    """
    Calculates cell size of given grid

    Parameters
    ----------
    x_max : numpy.ndarray
        Array of grid borders

    Returns
    -------
    dx : numpy.ndarray
        Array with size of cells
    """
    return np.append([0], np.diff(x_max, axis=0))


def get_y_i(x_max: numpy.ndarray, y_max: numpy.ndarray) -> numpy.array:
    """
    Calculates cell density values from given cumulative y array

    Parameters
    ----------
    x_max : numpy.ndarray
        Array of grid borders
    y_max: numpy.ndarray
        Array of cumulative distribution values

    Returns
    -------
    y_i : numpy.ndarray
        Array of density values
    """
    return np.append([0], np.where(np.diff(y_max) >= 0, np.diff(y_max) / np.diff(x_max), 0))


def get_dy_i(y_max: numpy.array) -> numpy.array:
    """
    Calculates population of each cell in a cumulative y array

    Parameters
    ----------
    y_max : numpy.array
        Cumulative y array

    Returns
    -------
    dy : numpy.ndarray
        Population array
    """
    return np.append(y_max[0], np.diff(y_max, axis=0))


def get_x_max(x):
    """
    Calculates grid from average x values assuming a regular grid. (not recommended! Better provide grid)

    Parameters
    ----------
    x : numpy.ndarray
        Average x values of grid cells

    Returns
    -------
    x_max : numpy.ndarray
        Grid array
    """
    x_max = np.zeros(len(x))
    for i in range(1, len(x_max)):
        x_max[i] = 2 * x[i] - x_max[i - 1]

    return x_max


class Distribution:
    def __init__(self,
                 x_max: numpy.ndarray | None = None,
                 y: numpy.ndarray | None = None,
                 y_max: numpy.ndarray | None = None,
                 base: int = 0,
                 x_type: int = 3,
                 d_f: list[float] | None = None,
                 time: float = 0.0,
                 norm_value: float | None = None,
                 **kwargs):
        """
        Create a Distribution object given either x_max and y_max or x_max and y

        Parameters
        ----------
        x_max : numpy.ndarray | None
            Grid of discrete distribution.
        y : numpy.ndarray | None
            Density of discrete distribution
        y_max : numpy.ndarray | None
            Cumulative values of discrete distribution
        base : int, optional
            Base of distribution. 0: number (default), 1: length, 2: surface, 3: volume
        x_type : int, optional
            Distributed quantity. 0: number, 1: length, 2: surface, 3: volume (default)
        d_f : list[float], optional
            List of fractal dimensions for length, surface and volume
        time : float, optional
            Time stamp of distribution. The default assumes initial state (time = 0.0)
        norm_value : float, optional
            Reference value for a normalized distribution quantity x
        kwargs : dict
            Dump for other keyword arguments
        """
        # set grid
        x_max = x_max if isinstance(x_max, np.ndarray) and not None else np.array(x_max)

        if y_max is not None:  # set distribution density if cumulative values are present
            y_max = y_max if isinstance(y_max, np.ndarray) else np.array(y_max)
            self.x_max, y_max = alg.set_x_0(x_max, y_max)
            self.x = get_x_i(self.x_max)
            self.y = get_y_i(self.x_max, y_max)
        elif y is not None:  # set distribution density
            y = y if isinstance(y, np.ndarray) else np.array(y)
            y_max = np.cumsum(np.multiply(y, get_dx_i(x_max)))
            # correct discretization error
            y_max /= y_max[-1]
            self.x_max, y_max = alg.set_x_0(x_max, y_max)
            self.x = get_x_i(self.x_max)
            self.y = y
        else:
            io_node.log_and_print('Insufficient data for distribution! Expected x_max and y_max or y!',
                                  kind='error')

        if base in range(4):
            self.base = base
        else:
            raise exc.WrongValue(expected=f'base to be in {range(4)}', received=base)

        if x_type in range(4):
            self.x_type = x_type
            self.x_name = x_types[x_type]
        else:
            raise exc.WrongValue(expected=f'x_type to be between 0 and {len(x_types) - 1}', received=x_type)

        if d_f is None:
            self.d_f = [0., 1., 2., 3.]
        else:
            self.d_f = d_f

        self.time = time

        self.normValue = norm_value
        if norm_value is not None:
            warnings.warn('[WARNING] Initial distribution with normalized values assumed!')

    def __str__(self, mode=0):
        return f'Distribution {DistributionNames[0]}' + r'$_{' + f'{self.base}' + r'}$(' + f'{x_types[self.x_type]})' \
            + f' with x_max: {np.power((self.x_max[1], self.x_max[-1]), float(1 / self.x_type))} µm on {len(self.x)} nodes'

    def __sub__(self, other):
        if self.AreCompatible(other):
            return Distribution(self.x, self.y - other.y, base=self.base, x_type=self.x_type, time=self.time)
        else:
            raise exc.IncompatibleDistributions(self, other)

    def __mul__(self, other):
        if self.AreCompatible(other):
            return Distribution(self.x, self.y * other.y, base=self.base, x_type=self.x_type, time=self.time)
        else:
            raise exc.IncompatibleDistributions(self, other)

    def __truediv__(self, other):
        if self.AreCompatible(other):
            newy = 1 * self.y
            othery = 1 * other.y
            newy = np.where(othery != 0, np.divide(newy, othery), 0)
            newy = np.nan_to_num(newy, posinf=0)
            return Distribution(self.x, newy, base=self.base, x_type=self.x_type, time=self.time)
        else:
            raise exc.IncompatibleDistributions(self, other)

    def __pow__(self, other):
        if self.AreCompatible(other):
            return Distribution(self.x, self.y ** other.y, base=self.base, x_type=self.x_type, time=self.time)
        else:
            raise exc.IncompatibleDistributions(self, other)

    def mix(self, *others, mass_frac: numpy.ndarray | list[float] | None = None):
        """
        Mixing distribution with other distributions according to mass fractions

        Parameters
        ----------
        others : tuple(Distribution)
            Distribution objects to be added
        mass_frac : numpy.ndarray | list[float] | None
            mass fractions of others in mix. Sum must be less than 1!
            default : equal mass fractions

        Returns
        -------
        Distribution
        """
        if mass_frac is None:
            # set equal mass fractions
            mass_frac = 1 / (len(others) + 1)
        else:
            if isinstance(mass_frac, list):
                mass_frac = np.array(mass_frac)
            if len(mass_frac) != len(others):
                io_node.log_and_print(
                    f'Wrong size of mass fractions. Expected {len(others)} but received {len(mass_frac)}',
                    kind='error')
                return self
            if mass_frac.sum() >= 1:
                io_node.log_and_print(
                    f'Sum of mass fractions must be < 1! Current sum = {mass_frac.sum()}',
                    kind='error')
                return self
        new_pop = (1 - mass_frac.sum()) / self.get_moment(1) * self.get_population()
        if self.x_type == 3 and self.base == 0:
            for idx, dist in enumerate(others):
                if all(dist.x == self.x) and self.base == dist.base:
                    new_pop += mass_frac[idx] / dist.get_moment(1) * dist.get_population()

            new_pop *= self.get_moment(1)
        temp_dist = self.copy()
        temp_dist.y = get_y_i(self.x_max, np.cumsum(new_pop))

        return temp_dist.change_base(self.base)

    def copy(self):
        return copy.deepcopy(self)

    def set_x_max(self, x_max: numpy.ndarray):
        """
        Sets new grid to distribution

        Parameters
        ----------
        x_max : numpy.ndarray
            Grid of distributed quantity

        Returns
        -------
        None
        """
        self.x_max = x_max.copy()
        self.x = get_x_i(x_max)

    def get_dict(self):
        """
        Returns a dictionary representation of the distribution object
        Returns
        -------
        dump_dict : dict
            dictionary representation of the distribution object
        """
        dump_dict = {
            'x_max': list(self.x_max),
            'y_max': list(self.get_y_max()),
            'base': self.base,
            'x_type': self.x_type,
            'd_f': self.d_f,
            'time': self.time,
            'normValue': self.normValue
        }
        return dump_dict

    def get_max_volume(self):
        """
        Get volume of x_95
        Returns
        -------
        x_95 : float
            x_95 value of distribution
        """
        y_max = self.change_base_and_x_type(3, 1).get_y_max()
        tmp_arr = np.zeros(len(y_max))
        tmp_arr[y_max <= 0.95] = 1
        return self.x_max[int(tmp_arr.sum())]

    def get_dx_i(self):
        return get_dx_i(self.x_max)

    def get_population(self):
        """
        Get population of each cell
        Returns
        -------
        population : numpy.ndarray
        """
        return self.y * self.get_dx_i()

    def get_y_max(self, message: bool = False):
        """
        Get cumulative cell values

        Parameters
        ----------
        message : bool, optional
            Print message about operation? default = False

        Returns
        -------
        y_max : numpy.ndarray
            Cumulative cell values
        """
        if message:
            io_node.log_and_print(f'Calculating Q{self.base}({self.x_type})')

        dy = self.get_population()
        return np.cumsum(dy)

    def adapt_grid(self, other):
        """
        Adapts grid of another distribution

        Parameters
        ----------
        other : Distribution
            Reference Distribution object

        Returns
        -------
        new_dist : Distribution
            Distribution with referenced grid
        """
        self.x_i_old = self.x.copy()
        self.x_max_old = self.x_max.copy()
        self.y_i_old = self.y.copy()
        return self.discretize(x_max=other.change_base_and_x_type(base=self.base,
                                                                  x_type=self.x_type).x_max)

    def discretize(self, x_max: numpy.ndarray | None = None,
                   nodes: int = 60,
                   grid: str = 'log',
                   log_fit: bool = False,
                   message: bool = False,
                   debug: bool = False):
        """
        Discretize the distribution on a new grid

        Parameters
        ----------
        x_max : numpy.ndarray, optional
            Left and right grid edges (min and max value for x). If None, keeps current x range
        nodes : int, optional
            number of cells, default = 60
        grid : str, optional
            Grid scale. 'log' = logarithmic scale (default), 'linear' = linear scale
        log_fit : bool, optional
            False: use actual values for linear interpolation (default), True: use log10 values for interpolation
        message : bool, optional
            Print message about operation? default = False
        debug : bool, optional
            Print results for debugging (default = False)

        Returns
        -------
        new_dist : Distribution
            New Distribution object with altered grid
        """
        if message:
            io_node.log_and_print('Discretizing distribution over {} nodes.'.format(nodes))
        new_dist = self.copy()
        limits = np.zeros(2)
        if x_max is None:
            # keep number of nodes but create a new grid within current x limits
            if self.x_max[0] > 0:
                limits[0] = self.x_max[0]
            elif len(self.x_max) > 2:
                limits[0] = self.x_max[1]
            else:
                limits[0] = self.x_max[1] * 0.01
                io_node.log_and_print(f"lower limit for discretization can't be 0! Set to {limits[0]}", kind='warning')

            limits[1] = self.x_max[-1]
        else:
            limits[0] = x_max[0]
            limits[1] = x_max[-1]
        x_max = alg.discretize_x(x_lim=limits, nodes=nodes, grid=grid)

        # interpolate values onto new grid
        if log_fit:
            f = interpol.interp1d(np.log10(self.x_max), self.get_y_max(),
                                  kind='linear', assume_sorted=True)
            y_max = f(np.log10(x_max[1:-1]))
        else:
            f = interpol.interp1d(self.x_max, self.get_y_max(),
                                  kind='linear', assume_sorted=True,
                                  fill_value=(0, 1), bounds_error=False)
            y_max = f(x_max[1:-1])
        y_max = np.append(np.append([0], y_max), [1])
        new_dist.y = get_y_i(x_max, y_max)
        new_dist.x = get_x_i(x_max)
        new_dist.x_max = x_max.copy()

        if debug:
            io_node.log_and_print(f'x = {new_dist.x[1]}')

            dfOld = pd.DataFrame({'x': self.x, 'init': self.y})
            dfNew = pd.DataFrame({'x': new_dist.x, 'interpol': new_dist.y})
            dfExport = pd.concat([dfOld, dfNew], axis=1)
            dfExport.to_csv(path_or_buf='Interpol_Dens.csv', sep=';', header=True,
                            index=False)
            dfOld = pd.DataFrame({'x': self.x_max, 'init': self.get_y_max()})
            dfNew = pd.DataFrame({'x': new_dist.x_max, 'interpol': new_dist.get_y_max()})
            dfExport = pd.concat([dfOld, dfNew], axis=1)
            dfExport.to_csv(path_or_buf='Interpol_Sum.csv', sep=';', header=True,
                            index=False)
        return new_dist

    def change_base_and_x_type(self, base: int = 0, x_type: int = 3, message: bool = False):
        """
        Change base and x type of distribution

        Parameters
        ----------
        base : int, optional
            Base of distribution. 0: number (default), 1: length, 2: surface, 3: volume
        x_type : int, optional
            Distributed quantity. 0: number, 1: length, 2: surface, 3: volume (default)
        message : bool, optional
            Print message about operation? default = False

        Returns
        -------
        new_dist : Distribution
            New Distribution object with altered base and x_type
        """

        return self.change_base(base=base, message=message).change_x_type(x_type=x_type, message=message)

    def change_base(self, base: int = 0, message: bool = False):
        """ Change base of distribution

        Parameters
        ----------
        base : int, optional
            Base of distribution. 0: number (default), 1: length, 2: surface, 3: volume
        message : bool, optional
            Print message about operation? default = False

        Returns
        -------
        new_dist : Distribution
            New Distribution object with altered base
        """
        new_dist = self.copy()
        if new_dist.base == base:
            if message:
                io_node.log_and_print('no need to change base, ignoring command', kind='info')
            return new_dist
        else:
            if message:
                io_node.log_and_print('Transforming from {}{}({}) to {}{}({})'.format(DistributionNames[0], self.base,
                                                                                      x_types[self.x_type],
                                                                                      DistributionNames[0], base,
                                                                                      x_types[self.x_type]))
            # calculate dx
            dxi = new_dist.get_dx_i()
            # calculate x^(s-r) * qr(x) * dx
            xqdx = np.power(new_dist.change_x_type(1).x[1:], float(self.d_f[base] - self.d_f[new_dist.base]))
            xqdx = np.append([0], xqdx)
            xqdx *= new_dist.get_population()

            # normalize by integral for x from 0 to inf
            xqdx = xqdx / np.sum(xqdx)

            new_dist.y = xqdx[1:] / dxi[1:]
            new_dist.y = np.append([0], new_dist.y)
            # update base
            new_dist.base = base
            return new_dist

    def change_x_type(self, x_type: int, message: bool = False):
        """ Change base and x type of distribution
        Parameters
        ----------
        x_type : int, optional
            Distributed quantity. 0: number, 1: length, 2: surface, 3: volume (default)
        message : bool, optional
            Print message about operation? default = False

        Returns
        -------
        new_dist : Distribution
            New Distribution object with x_type
        """
        if self.x_type == x_type:
            if message:
                io_node.log_and_print('no need to change x_type, ignoring command', kind='info')
            return self.copy()
        else:
            if message:
                io_node.log_and_print('Transforming {}{}({}) to {}{}({})'.format(DistributionNames[0], self.base,
                                                                                 x_types[self.x_type],
                                                                                 DistributionNames[0],
                                                                                 self.base, x_types[x_type]))
            x_max_new = np.zeros(len(self.x_max))
            if self.x_type == 1:
                if x_type == 3:
                    # get volume from size
                    x_max_new = np.pi / 6.0 * np.power(self.x_max,
                                                       float(self.d_f[x_type]) / float(self.d_f[self.x_type]))
                else:
                    io_node.log_and_print(
                        f'Change from {x_types[self.x_type]} to {x_types[x_type]} not implemented yet!')
            elif self.x_type == 3:
                if x_type == 1:
                    # get size from volume
                    x_max_new = np.power(6.0 * self.x_max / np.pi,
                                         float(self.d_f[x_type]) / float(self.d_f[self.x_type]))
                else:
                    io_node.log_and_print(
                        f'Change from {x_types[self.x_type]} to {x_types[x_type]} not implemented yet!')
                """ 
                ToDo: 
                implement length to surface change
                """

            new_dist = self.copy()
            new_dist.x = get_x_i(x_max_new)
            new_dist.x_max = x_max_new
            new_dist.y = get_y_i(new_dist.x_max, self.get_y_max())
            new_dist.x_type = x_type
            return new_dist

    def get_moment(self, k: int = 0, x_min: float = 0.):
        """
        Calculate kth moment of distribution

        Parameters
        ----------
        k : int, optional
            kth moment to be calculated. Default is 0th moment (integral of density distribution)
        x_min : float, optional
            Lower limit for moment calculation
        Returns
        -------
        moment : float
            kth moment from x_min to infinity
        """
        if x_min > 0:
            f = interpol.interp1d(self.x_max, self.get_y_max())
            x_range = np.append([x_min], self.x_max[self.x_max > x_min])
            x_i = np.diff(x_range)
            limits = f(x_range)
            mass_frac = np.diff(limits)
            return np.sum(np.power(x_i, k) * mass_frac)
        else:
            xkqdx = np.power(self.x, k) * self.get_population()
            return np.sum(xkqdx)

    def get_DI(self, x_limit: float = np.inf):
        """
        Calculate dispersion index as int(q_3(x), x=0..x_limit) / x_50,3(x_limit)

        Parameters
        ----------
        x_limit : float, optional
            Upper limit for DI calculation. Default takes whole distribution
        Returns
        -------
        DI : float
            Dispersion index
        """
        # get mass distribution over size plot
        new_dist = self.change_x_type(x_type=1)
        new_dist = new_dist.change_base(base=3)

        f = interpol.interp1d(x=new_dist.get_y_max(), y=new_dist.x_max)
        A_peak = (new_dist.get_population()[new_dist.x_max <= x_limit]).sum()
        x_50 = f(A_peak / 2)

        return A_peak / x_50

    def get_x_50(self, x_limit: float = np.inf):
        """
        Get median of distribution

        Parameters
        ----------
        x_limit : float, optional
            Upper limit for distributed quantity

        Returns
        -------
        x_50 : float
            Median of distribution
        """
        f = interpol.interp1d(x=self.get_y_max(), y=self.x_max)
        A_peak = (self.get_population()[self.x_max <= x_limit]).sum()
        return f(A_peak / 2)

    def get_x_of_percentile(self, percentile: float | Iterable):
        """
        Get value of nth percentile

        Parameters
        ----------
        percentile : float | Iterable
            Desired percentiles to be calculated
        Returns
        -------
        x : numpy.ndarray
            x values of requested percentiles
        """
        f = interpol.interp1d(x=self.get_y_max(), y=self.x_max)
        return f(percentile)

    def get_percentile_of_x(self, x: float):
        """
        Get fraction of population <= x

        Parameters
        ----------
        x : float
            Value of distributed quantity
        Returns
        -------
        percentile : float
            Fraction of population with <= x
        """
        if x >= self.x_max[-1]:
            return 1
        else:
            f = interpol.interp1d(x=self.x_max, y=self.get_y_max())
            return f(x)

    def get_agg_volumes(self):
        """
        Returns a matrix of all possible aggregate volumes of particles in grid
        Returns
        -------
        ndarray
            matrix of all possible aggregate volumes
        """
        I_max = len(self.x)  # grid size
        return np.reshape(self.x, (I_max, 1)) + np.reshape(self.x, (1, I_max))

    def get_agg_A_ijk(self):
        """
        Get all cell combination within limits of new particle cell
        Returns
        -------
        A_ijk : numpy.ndarray
            3D matrix with all possible combinations of cell values for every cell
        """

        x = self.x.copy()  # discrete particle size
        x_max = self.x_max.copy()  # cell limits
        I_max = len(x)  # grid size
        volume_sum = self.get_agg_volumes()  # sum of representative volumes
        agg_A_ijk = np.zeros((I_max, I_max, I_max))
        for i in range(I_max):
            jk_matrix = np.where(np.logical_and(volume_sum >= x_max[i - 1], volume_sum < x_max[i]), 1, 0)
            # select only j >= k by taking lower diagonal
            agg_A_ijk[i] = np.tril(jk_matrix)

        return agg_A_ijk

    def AreCompatible(self, other):
        """
        Checks if other is compatible with Distribution object. Grid, base and x_type are to be equal

        Parameters
        ----------
        other : Distribution
            Other Distribution object

        Returns
        -------
        bool
        """
        if self.base == other.base and self.x_type == other.x_type:
            return True
        else:
            return False

    def plot(self, mode: int = 0,
             population: bool = True,
             x_type: int | None = None,
             base: int | None = None,
             **plot_kwargs):
        """
        Plots distribution

        Parameters
        ----------
        mode : int, optional
            0 = density distribution (default), 1 = cumulative distribution
        population : bool, optional
            If mode = 0, then returns population (default = True) or density (False)
        base : int, optional
            Base of distribution. 0: number (default), 1: length, 2: surface, 3: volume
        x_type : int, optional
            Distributed quantity. 0: number, 1: length, 2: surface, 3: volume (default)
        plot_kwargs : dict, optional
            Plot arguments (see io_node.plot_on_ax)

        Returns
        -------
        Figure, Axes
        """
        if base is None:
            base = self.base
        if x_type is None:
            x_type = self.x_type
        fig, ax = plt.subplots()
        self.plot_on_ax(ax=ax, mode=mode, population=population, x_type=x_type, base=base, **plot_kwargs)
        ax.set_xlabel = f'{x_types[x_type]} / -'

        if population:
            ax.set_ylabel = r'$f_i(' + x_types[x_type] + '_i)$ / -'
        else:
            ax.set_ylabel = r'$f(' + x_types[x_type] + ')$ / -'
        return fig, ax

    def plot_on_ax(self, ax, mode: int = 0, population: bool = True,
                   x_type: int | None = None, base: int | None = None, **plot_kwargs):
        """
        Plots distribution on given ax

        Parameters
        ----------
        ax : Axes
            Pyplot.Ax object for the data plot
        mode : int, optional
            0 = density distribution (default), 1 = cumulative distribution
        population : bool, optional
            If mode = 0, then returns population (default = True) or density (False)
        base : int, optional
            Base of distribution. 0: number, 1: length, 2: surface, 3: volume
            default : object base attribute
        x_type : int, optional
            Distributed quantity. 0: number, 1: length, 2: surface, 3: volume
            default : object base attribute
        plot_kwargs : dict, optional
            Plot arguments (see io_node.plot_on_ax)

        Returns
        -------
        None
        """
        if base is None:
            base = self.base
        if x_type is None:
            x_type = self.x_type
        # modify distribution
        new_dist = self.copy()
        new_dist = new_dist.change_base_and_x_type(base=base, x_type=x_type)
        if mode == 0:
            x_plot = new_dist.x.copy()
            if population:
                y_plot = new_dist.get_population()
            else:
                y_plot = new_dist.y.copy()
        elif mode == 1:
            x_plot = new_dist.x_max.copy()
            y_plot = new_dist.get_y_max()
        else:
            io_node.log_and_print(f'Wrong plot mode! Expected: 0 or 1, received: {mode}\n'
                                  f'Only 0: density/population and 1: cumulative allowed!', kind='error')
            x_plot = np.zeros(2)
            y_plot = np.zeros(2)

        # param_set plot_kwargs
        plot_kwargs.setdefault('label',
                               rf'$f_{new_dist.base}({x_types[new_dist.x_type]}, {np.round(new_dist.time, 2)} s)$')
        plot_kwargs.setdefault('x_log', True)
        plot_kwargs.setdefault('y_log', False)
        io_node.plot_on_ax(x_plot=x_plot, y_plot=y_plot, ax=ax, **plot_kwargs)
        if plot_kwargs['x_log']:
            ax.set_xscale('log')
        if plot_kwargs['y_log']:
            ax.set_yscale('log')

    def to_csv(self, path: str):
        if not path.endswith('.csv'):
            if not path.endswith('/'):
                path += '/'
            path += f'dist_q{self.base}({x_names[self.x_type]}, {self.time}).csv'
        df = pd.DataFrame()
        df[f'x'] = self.x
        df[f'y'] = self.y
        df[f'x_max'] = self.x_max
        df[f'y_max'] = self.get_y_max()
        df.to_csv(path_or_buf=path, sep=';', header=True, index=False)


def get_distribution(kind: str = 'custom', **kwargs):
    """
    returns an analytical distribution

    Parameters
    ----------
    kind : str
        'mono', 'dec_exponential', 'inc_exponential', 'gauss', 'log-normal' or 'custom' (default)
    kwargs :
        keyword arguments to be passed to Distribution

    Returns
    -------

    """
    if kind == 'mono':
        return mono(**kwargs)
    elif kind == 'dec_exponential':
        return dec_exponential(**kwargs)
    elif kind == 'inc_exponential':
        return inc_exponential(**kwargs)
    elif kind == 'gauss':
        return gauss(**kwargs)
    elif kind == 'log-normal':
        return log_normal(**kwargs)
    else:
        if 'x_max' in kwargs.keys():
            return Distribution(**kwargs)


def import_psd(path=None, base: int = 0, x_type: int = 3, d_f: list | None = None,
               time: list | float = 0, norm_value: float | None = None,
               dist_adapt_to: Distribution | None = None,
               string_key: bool = False, in_dict: bool = True,
               number_based: bool = True,
               **rest_args):
    """
    Imports data from a CSV or XLSX file and returns is as a Distribution object

        Parameters
    ----------

    path : str
        Path to file
    base : int, optional
            Base of distribution. 0: number (default), 1: length, 2: surface, 3: volume
    x_type : int, optional
        Distributed quantity. 0: number, 1: length, 2: surface, 3: volume (default)
    d_f : list[float], optional
        List of fractal dimensions for length, surface and volume
    time : float, optional
        Time stamp of distribution. The default assumes initial state (time = 0.0)
    norm_value : float, optional
        Reference value for a normalized distribution quantity x
    kwargs : dict
        Dump for other keyword arguments
    dist_adapt_to : Distribution
        adapts Distributions grid to a given reference
    in_dict : bool
        True (default) returns all imported Distributions as a dictionaries with
    string_key : bool
        True (default) converts all dictionary keys to strings
    number_based : bool
        False (default) automatically converts distributions to number based (base = 0) volume (x_type = 3) distribution
    rest_args : dict
        catching all other kwargs

    Returns
    -------

    """
    if path is not None:
        dist_dict = {}
        df_dict = {}

        if path.endswith('.csv'):  # check if file is a CSV
            try:
                df = pd.read_csv(path, sep=';', header=0)
            except FileNotFoundError:
                df = pd.read_csv(path, sep=';', header=0)
            df_dict = {time: df}

        elif path.endswith('.xlsx'):  # check if file is a XLSX
            # read excel
            xls = pd.ExcelFile(path)
            df_dict = {str(name) if string_key else float(name): xls.parse(name) for name in xls.sheet_names}

        for time, df_run in df_dict.items():
            # extract data
            if 'x_max' in df_run.columns:
                x_max = df_run['x_max'].to_numpy()
            else:
                io_node.log_and_print(f'Insufficient data to import {path}. Expected x_max grid data!', kind='error')
                return None

            if 'y_max' in df_run.columns:
                y_max = df_run['y_max'].to_numpy()
            elif 'y_pop' in df_run.columns:
                y_max = np.cumsum(df_run['y_pop'].to_numpy())
            else:
                io_node.log_and_print(f'Insufficient data to import {path}. Expected y_max or y_pop data!',
                                      kind='error')
                return None

            # create new distribution
            if d_f is None:
                d_f = [0, 1, 2, 3]  # default for fractal dimension
            new_dist = Distribution(x_max=x_max, y_max=y_max, base=base, x_type=x_type,
                                    norm_value=norm_value, time=time, d_f=d_f)
            if dist_adapt_to is not None:
                new_dist = new_dist.adapt_grid(dist_adapt_to)

            # convert to number based volume distribution
            if number_based:
                new_dist = new_dist.change_base_and_x_type(base=0, x_type=3)

            dist_dict[time] = new_dist.copy()

        if not in_dict:
            if len(dist_dict) > len(df_dict):
                io_node.log_and_print(f'Only one of {len(df_dict)} Sheets imported.\n'
                                      f'Sheet name: {time} \n'
                                      f'Consider using in_dict=False (default) for full output.',
                                      kind='warning')
            return dist_dict[time]
        else:
            return dist_dict
    else:
        io_node.log_and_print(f"path_in can't be None", kind='error')


def mix(*dists, mass_frac: numpy.ndarray | None = None):
    """
    mix distributions according to mass fractions

    Parameters
    ----------
    dists : tuple[Distribution]
        any number of Distribution objects
    mass_frac : numpy.ndarray | None
        mass fractions of distributions

    Returns
    -------
    Distribution
        mixed Distribution
    """
    return dists[0].mix(*dists[1:], mass_frac=mass_frac[1:])

def mono(nodes: int = 30,
         x_lim: numpy.ndarray = None,
         peak_at: float = 1.,
         base: int = 0,
         x_type: int = 3,
         **rest_args):
    """

    Parameters
    ----------
    nodes
    x_lim
    peak_at
    base
    x_type
    rest_args

    Returns
    -------

    """
    # param_set distribution to mono at x=peak
    x_max = alg.discretize_x(x_lim=np.array(x_lim), nodes=nodes + 1, grid=rest_args.setdefault('grid', 'log'))
    x_max = alg.set_x_0(x_max)
    if x_max[0] < peak_at <= x_max[-1]:
        y_max = np.zeros(len(x_max))
        y_max[(x_max < peak_at).sum():] = 1
        dist = Distribution(x_max=x_max, y_max=y_max, base=base, x_type=x_type)
        return dist
    else:
        io_node.log_and_print(f"peak outside of range! x_lim: {x_lim}, peak_at: {peak_at}", kind="warning")


def dec_exponential(nodes=30,
                    x_lim=None,
                    N_0=1,
                    v_0=1,
                    base: int = 0,
                    x_type: int = 3,
                    **rest_args):
    # param_set distribution to mono at x=peak
    x_max = alg.discretize_x(x_lim=np.array(x_lim), nodes=nodes + 1, grid=rest_args.setdefault('grid', 'log'))
    x_max = alg.set_x_0(x_max)
    n = lambda v: N_0 / v_0 * np.exp(-v / v_0)
    return Distribution(x_max=x_max, y=n(get_x_i(x_max)), base=base, x_type=x_type)


def inc_exponential(nodes=30,
                    x_lim: list | None = None,
                    N_0: float = 1,
                    v_0: float = 1,
                    base: int = 0,
                    x_type: int = 3,
                    **rest_args):
    # param_set distribution to mono at x=peak
    x_max = alg.discretize_x(x_lim=np.array(x_lim), nodes=nodes + 1, grid=rest_args.setdefault('grid', 'log'))
    x_max = alg.set_x_0(x_max)
    n = lambda v: N_0 / v_0 * np.exp(v / v_0)
    return Distribution(x_max=x_max, y=n(get_x_i(x_max)), base=base, x_type=x_type)


def gauss(nodes: int = 30,
          x_lim: list | None = None,
          N_0: float = 1,
          mu: float = 1,
          sigma: float = np.sqrt(np.log(4 / 3)),
          base: int = 0,
          x_type: int = 3,
          **rest_args):
    """

    Parameters
    ----------
    nodes : int
        number of nodes, default = 30
    x_lim : list | None
        x range, default = [0, 2]
    N_0 : float
        total integral, default = 1
    mu : float
        default: 1
        Mean of distribution
    sigma : float
        standard deviation of distribution
        default: np.sqrt(np.log(4 / 3))
    base : int, optional
            Base of distribution. 0: number (default), 1: length, 2: surface, 3: volume
    x_type : int, optional
        Distributed quantity. 0: number, 1: length, 2: surface, 3: volume (default)
    rest_args

    Returns
    -------

    """
    if x_lim is None:
        x_lim = [0, 2]
    # param_set distribution to gauss
    x_max = alg.discretize_x(x_lim=np.array(x_lim), nodes=nodes + 1, **rest_args)
    x_max= alg.set_x_0(x_max)
    n = lambda v: N_0 / (np.sqrt(2 * np.pi) * v * sigma) * np.exp(-np.power(np.log(v / mu), 2) / (2 * sigma ** 2))
    return Distribution(x_max=x_max, y=np.append([0], n(get_x_i(x_max)[1:])), base=base, x_type=x_type)


def log_normal(x_lim: list[float],
               nodes: int = 30,
               grid: str = 'log',
               log_base: float = 10.,
               mu: float = 1,
               sigma: float = np.sqrt(np.log(4 / 3)),
               base: int = 0,
               x_type: int = 3,
               **dist_kwargs):
    """
    lognormal distribution as Distribution Object

    Parameters
    ----------
    x_lim: list[float] or None
        x limits as list with at least two floats sorted in ascending order.
        First and last item represent x_min and x_max respectively
    nodes: int, optional
        number of grid points (default = 30)
    grid: str, optional
        'log': (default) logarithmic grid
        'linear': linear grid
    log_base: float, optional
        default: 10
        Base of logarithmic grid.
    mu : float, optional
        default: 1
        Mean of distribution
    sigma : float, optional
        standard deviation of distribution
        default: np.sqrt(np.log(4 / 3))
    base : int, optional
            Base of distribution. 0: number (default), 1: length, 2: surface, 3: volume
    x_type : int, optional
        Distributed quantity. 0: number, 1: length, 2: surface, 3: volume (default)
    dist_kwargs: dict
        kwargs for Distribution object

    Returns
    -------

    """
    # get grid border points
    x_max = alg.discretize_x(x_lim=np.array(x_lim), nodes=nodes + 1, grid=grid, grid_base=log_base)
    x_max = alg.set_x_0(x_max)
    return Distribution(x_max=x_max, y_max=lognorm.cdf(x_max, s=sigma, scale=mu), base=base, x_type=x_type,
                        **dist_kwargs)