import numpy
import pandas

from src import io_node


def set_x_0(x: numpy.ndarray, y: numpy.ndarray | None = None):
    """
    Sets first value in x array to 0

    Parameters
    ----------
    x : numpy.array
        x values of distribution
    y : numpy.ndarray | None
        y values of distribution. Zero assumed if None

    Returns
    -------
    x : numpy.ndarray, y: numpy.ndarray
        new values for x and y
    """

    if y is None:
        x.sort()
        if x[0] != 0:
            return numpy.append([0], x)
        else:
            return x
    else:
        y = y[x.argsort()]
        x.sort()
        if x[0] != 0:
            return numpy.append([0], x), numpy.append([0], y if y is not None else numpy.zeros(len(x)))
        else:
            return x, y if y is not None else numpy.zeros(len(x))


def log_of_base(x: numpy.ndarray, base: float) -> numpy.ndarray:
    """
    calculate logarithm of an arbitrary base

    Parameters
    ----------
    x : numpy.ndarray
        array with initial values
    base : float
        logarithm base b as in x = b^log_b(x)

    Returns
    -------
    numpy.ndarray
    """
    return numpy.log(x) / numpy.log(base)


def discretize_x(x_lim: numpy.ndarray | list, nodes: int = 60, grid: str = 'log', grid_base: float = 10., debug: bool = False,
                 **rest_args) -> numpy.ndarray:
    """
    discretize x on a defined grid

    Parameters
    ----------
    x_lim: numpy.ndarray
        limits of x sorted in ascending order. At least two values required
    nodes: int
        number of cells
    grid: str
        'log': logarithmic grid with x_{i+1} = x_i * a, default
        'linear': linear grid
        'geom': geometric grid with x_i = grid_base^i
    grid_base: float
        base of geometric grid
    debug: bool
        default: True
    rest_args

    Returns
    -------
    numpy.ndarray
    """
    if not isinstance(x_lim, numpy.ndarray):
        x_lim = numpy.array(x_lim)
    x_lim.sort()
    if debug:
        io_node.log_and_print('xmin = {}, xmax = {}'.format(x_lim[0], x_lim[-1]))
    # create new x grid
    if grid == 'linear':
        new_x = numpy.linspace(x_lim[0], x_lim[-1], num=nodes)
    elif grid == 'log':
        new_x = numpy.linspace(log_of_base(x_lim[0], grid_base), log_of_base(x_lim[1], grid_base), num=nodes)
        new_x = numpy.power(grid_base, new_x)
    elif grid == 'geom':
        # increase number of nodes to fit x_lim
        if x_lim[-1] > x_lim[0] * numpy.power(grid_base, nodes):
            nodes = int(numpy.ceil(log_of_base(x_lim[-1] / x_lim[0], grid_base)))
        new_x = numpy.geomspace(x_lim[0], x_lim[0] * numpy.power(grid_base, nodes), num=nodes)
    else:
        new_x = None
    return new_x


def split_df_by_bounds(df: pandas.DataFrame, bounds: list[float], x_name: str | None = None):
    """
    Splits DataFrame by given bounds

    Parameters
    ----------
    df : list[float]
        DataFrame with data to be split in rows according a reference column
    bounds : list[float]
        upper limits for values in reference column
    x_name : str | None, optional
        name of the reference column, with first column of DataFrame as default.

    Returns
    -------
    list[pandas.DataFrames]
        list of DataFrames split by bounds along reference column
    """
    if x_name is None:
        x_name = df.columns[0]
    dfs = []
    for i in range(len(bounds)):
        if i == 0:
            dfs.append(df[df[x_name] <= bounds[i]])
        elif i == len(bounds):
            dfs.append(df[df[x_name] > bounds[i - 1]])
        else:
            dfs.append(df[(df[x_name] > bounds[i - 1]) & (df[x_name] <= bounds[i])])
    return dfs
