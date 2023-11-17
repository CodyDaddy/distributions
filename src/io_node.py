import matplotlib.pyplot
import numpy
import numpy as np
import pandas
from importlib import reload
from collections.abc import Callable
import json
from numpyencoder import NumpyEncoder

import matplotlib.pyplot as plt
import os
import fnmatch
from datetime import datetime
import time
import requests
import tkinter as tk
from tkinter import filedialog


from mpl import kit_colors as kit


def send_telegram_message(message, token, chat_id):
    """
    sends telegram message

    Parameters
    ----------
    message: str
        message text
    token: str
        bot token
    chat_id: str
        chat id

    Returns
    -------
    str
    response as json dict
    """
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    data = {'chat_id': chat_id, 'text': message}
    response = requests.post(url, data=data)
    return response.json()


def folder_picker(init_path):
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()
    return folder_path


def only_folders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def files_with_pattern(path, pattern):
    return [name for name in os.listdir(path) if
            os.path.isfile(os.path.join(path, name)) and fnmatch.fnmatch(name, pattern)]


def add_col_and_defragment(df, col_name, data):
    """
    adds column to DataFrame and defragments it

    Parameters
    ----------
    df : DataFrame
    col_name : str
        New column name
    data : ndarray

    Returns
    -------
    DataFrame
    """
    df[col_name] = data
    return df.copy()


def digits_forced(x: int, n: int = 2) -> str:
    """
    returns x as double digit string with '0x' if x is single digit number

    Parameters
    ----------
    x : int
        digit (month, day, hour or minute)
    n : int
        number of digits desired

    Returns
    -------
    x_string : str
        string of x with leading zeros
    """
    if x < 10 ** (n - 1):
        return '0' * (n - len(str(x))) + str(x)
    else:
        return str(x)


def get_time_stamp(p_date=True, p_time=True, sep=True):
    """
    Returns time stamp as string as {year}{month}{day}-{hour}{minute}

    Parameters
    ----------
    p_date: bool, optional
        use date in string (default=True)
    p_time: bool, optional
        use time in string (default=True)
    sep: bool, optional
        separate date and time with '-' (default=True)

    Returns
    -------

    """
    dt_stamp = datetime.now()
    stamp = ''
    if p_date:
        stamp += f'{dt_stamp.year}'
        stamp += digits_forced(dt_stamp.month)
        stamp += digits_forced(dt_stamp.day)
        if sep:
            stamp += '-'
    if p_time:
        stamp += digits_forced(dt_stamp.hour)
        stamp += digits_forced(dt_stamp.minute)
    return stamp


def runtime_test(func: Callable):
    """
    runs a runtime test on a callable returns its output

    Parameters
    ----------
    func

    Returns
    -------

    """
    # print runtime of function
    start = time.time()
    return_obj = func
    end = time.time()
    print(f'RUNTIME of {func.__name__}: {start - end} s')
    return return_obj


def log_and_print(message, print_terminal=True, logger_args: dict | None = None, kind='info'):
    if kind is not None:
        message = '[' + kind.upper() + '] ' + message
    if print_terminal:
        print(message)
    if logger_args is not None:
        if logger_args['output_log']:
            logger_args['logger'].info(message)


def plot_on_ax(x_plot: numpy.ndarray,
               y_plot: numpy.ndarray,
               ax: plt.Axes,
               c: str | tuple = kit.black,
               lw: float = 1.,
               linestyle: str = '-',
               marker: str | None = 'o',
               mfc: str | tuple | None = None,
               mec: str | tuple | None = None,
               ms: float = 6,
               mew: float = 1,
               alpha: float = 0.9,
               label: str = '#noname',
               x_log: bool = True,
               y_log: bool = False,
               grid: bool | dict = True,
               x_label: str | None = None,
               y_label: str | None = None,
               legend: bool = True,
               ncol: int = 1
               ):
    """
    Plots data on given Ax (beautiful plots by Marvin&Frank)

    Parameters
    ----------
    x_plot : numpy.ndarray
        x data
    y_plot : numpy.ndarray
        y data
    ax : Axes
        ax
    c : str | tuple
        line color, default = kit.black,
    lw: float
        line width, default = 1.,
    linestyle: str
        line style, default = '-',
    marker: str | None
        marker style, default = 'o',
    mfc: str | tuple | None
        marker face color, default = None,
    mec: str | tuple | None
        marker edge color, default = None,
    ms: float
        marker size, default = 6,
    mew: float
        marker edge width, default = 1,
    alpha: float
        transparency, default = 0.9,
    label: str
        legend data label, default = '#noname'
    x_log: bool
        logarithmic x-axis, default = True
    y_log: bool
        logarithmic y-axis, default= False,
    grid: bool | dict
        plot grid, default= True or {'which': 'major', 'axis': 'both'}
    x_label: str | None
        x-axis label, default= None,
    y_label: str | None
        y-axis label, default= None
    legend: bool
        plot legend, default = True,
    ncol: int
        number of columns in legend, default = 1

    Returns
    -------
    None
    """

    ax.plot(
        x_plot,  # x-data
        y_plot,  # y-data
        c=c,  # line color
        lw=lw,  # line width
        linestyle=linestyle,  # line options
        marker=marker,  # marker style
        mfc=mfc,  # marker face color
        mec=mec,  # marker edge color
        ms=ms,  # marker size
        mew=mew,  # marker edge width
        alpha=alpha,  # transparency
        label=label  # label
    )
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')

    ax.grid(grid)

    if legend:
        ax.legend(ncol=ncol)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)


def plot_df_fade_on_ax(ax: matplotlib.pyplot.Axes,
                       df: pandas.DataFrame,
                       x: str,
                       y: list,
                       c: str | tuple = kit.green,
                       x_lim: list | None = None,
                       y_lim: list | None = None,
                       x_log: bool = True,
                       y_log: bool = False,
                       grid: bool = True,
                       legend: bool = True,
                       ncol: int = 1,
                       **plot_kwargs):
    """
    Takes a Dataframe and plots its data column-wise as a faded curve over data from column x

    Parameters
    ----------
    ax : Axes
        Pyplot Axes objects
    df : DataFrame
        a DataFrame with data to be plotted
    x : str
        x column title
    y : list
        a list of y columns, if None then all columns except x are considered
    c : str | tuple
        color of the plot. Either hex string or rpg/cmyk tuple
    y_lim : list | numpy.ndarray
        y limits
    x_lim : list | numpy.ndarray
        x limits
    x_log : bool
        to plot on logarithmic x ax, default = True
    y_log : bool
        to plot on logarithmic y ax, default = True
    ncol : int
        number of columns in legend, default = 1
    legend : bool
        to plot legend, default = True
    grid : bool | dict
        to plot a grid or dict with grid kwargs, default = True or  {'which': major, 'axis': 'both'}
    plot_kwargs : plot arguments

    Returns
    -------
    None
    """
    x_plot = df[x].to_numpy()
    if y is None:
        y = df.loc[:, df.columns != 'x'].columns.values
    for idx, col in enumerate(y):
        y_plot = df[col].to_numpy()
        # get total birth mass
        if plot_kwargs.setdefault('base', 3) == 1:
            y_plot *= np.power(x_plot, 3)

        label = plot_kwargs.setdefault('label', '')
        c = kit.hex_to_faded(c, (idx + 1) / len(y))
        plot_on_ax(x_plot=x_plot, y_plot=y_plot,
                   ax=ax,
                   c=c,
                   mfc=c,
                   mec=c,
                   lw=plot_kwargs.setdefault('lw', 1),
                   linestyle=plot_kwargs.setdefault('linestyle', '-'),
                   marker=plot_kwargs.setdefault('marker', None),
                   alpha=plot_kwargs.setdefault('alpha', 0.9),
                   label=label + f'{np.round(col, 1) if isinstance(col, float) else col}')

        if plot_kwargs.setdefault('x_lim', None) is not None:
            ax.set_xlim(x_lim)
        else:
            ax.set_xlim(left=df.loc[1, x], right=df.loc[-1, x])
        if y_lim is not None:
            ax.set_ylim(left=y_lim[0], right=y_lim[-1])

        if x_log:
            ax.set_xscale('log')
        if y_log:
            ax.set_yscale('log')

        ax.grid(grid)

        if legend:
            ax.legend(ncol=ncol)


def plot_progress_on_ax(ax: matplotlib.pyplot.Axes,
                        df: pandas.DataFrame,
                        mark_init=True,
                        shadow: int = 5,
                        x_lim: list | None = None,
                        y_lim: list | None = None,
                        x_log: bool = True,
                        y_log: bool = False,
                        grid: bool = True,
                        legend: bool = True,
                        ncol: int = 1,
                        marker: str | None = None,
                        **plot_kwargs
                        ):
    """


    Parameters
    ----------
    ax
    df
    mark_init
    shadow
    x_lim
    y_lim
    x_log
    y_log
    grid
    legend
    ncol
    marker
    plot_kwargs

    Returns
    -------

    """
    cols = df.columns
    # plot initial state
    if mark_init:
        plot_on_ax(x_plot=df[cols[0]], y_plot=df[cols[1]], ax=ax, c=kit.blue, label=cols[1], marker=marker,
                   **plot_kwargs)

    # plot shadow
    if len(cols) - 2 > shadow:
        for i in range(shadow):
            idx = i - 1 - shadow
            plot_on_ax(x_plot=df[cols[0]], y_plot=df[cols[idx]], ax=ax, c=kit.gray((i + 1) / shadow), label=cols[idx],
                       marker=marker, **plot_kwargs)
    else:
        for i in range(2, len(cols) - 1):
            plot_on_ax(x_plot=df[cols[0]], y_plot=df[cols[i]], ax=ax, c=kit.gray((i - 2) / shadow), label=cols[i],
                       marker=None, **plot_kwargs)
    if len(cols) > 1:
        # plot state
        plot_on_ax(x_plot=df[cols[0]], y_plot=df[cols[-1]], ax=ax, c=kit.red, label=cols[-1], marker=None)

    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    else:
        ax.set_xlim(df[df.columns[0]].to_numpy()[1], df[df.columns[0]].to_numpy()[-1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    if x_log:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')
    if y_log:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')

    ax.grid(grid)

    if legend:
        ax.legend(ncol=ncol)
