# imports
from abc import ABC

import numpy
import pandas
import copy

from mpl import kit_colors
from src.viscosity_models import get_eta_herschel_bulkley, get_eta_herschel_bulkley_log, get_herschel_bulkley_fit
from src.algorithms import split_df_by_bounds
from src import io_node


class Base(ABC):
    def __init__(self, name: str = 'unnamed', kind: str = 'custom', params: dict | None = None,
                 func_type: str = 'step', **kwargs):
        """
        Basic class for input objects

        Parameters
        ----------
        name : str
            Name of the object. Default : 'unnamed'
        kind : str
            Abstract parameter that holds unique group of this object
            e.g. name of viscosity model, material group, kernel model etc.
        params : dict | None
            dictionary of parameter iterables with parameter name as string key.
            Usually with a variable 'x_max' holding limit values for a discrete parameter model.
            Example:
                Assume parameters 'a' and 'b' depend on property 'x'.
                params = {'x_max': [1, 2, 3, 4], 'a': [10, 20, 30, 40], 'b': [0.1, 0.2, 0.3]}
                For x<=1 returns model with parameters a=10, b=0.1,
                for 1<x<=2 returns model with parameters a=20, b=0.2, etc.
        func_type : str
            function type of parameters along variable axis.
            'step' : constant parameter within range (default)
            'linear' : linear parameter change along entity property
        kwargs
        """
        self.name = name
        self.kind = kind
        self.params = params
        self.func_type = func_type

    def copy(self):
        return copy.deepcopy(self)


class Viscosity(Base):
    def __init__(self, params: dict | None = None, kind: str = 'Newton', **kwargs):
        """
        Viscosity represented as object

        Parameters
        ----------
        params : dict | None
            dictionary of parameter iterables with parameter name as string key.
            With a variable 'x_max' holding limit values for a discrete parameter model
            for viscosity over shear rate.
        kind : str
            Name of viscosity model
        kwargs
        """
        super(Base, Viscosity).__init__(**kwargs)
        self.name += '_viscosity'

    def get_eta(self, gamma: float | numpy.ndarray | list):
        r"""
        Calculates dynamic viscosity for given shear rate(s) depending on self.kind and self.params

        Parameters
        ----------
        gamma : float | numpy.ndarray | list
            Shear rate(s)

        Returns
        -------
        eta : numpy.ndarray
            dynamic viscosity for given shear rate. self.kind holds model name.

            Model names :

            'Newton' : constant viscosity

            'Herschel-Bulkley' : Hershel-Bulkley model with

            .. math:: \eta = \left\{\begin{array}{ll}\eta_0, & \dot{\gamma} \leq \dot{\gamma}_0 \\ \eta_0 + k \dot{\gamma}^{n-1}, & \, \dot{\gamma} > \dot{\gamma}_0 \\ \end{array}\right.
        """
        if isinstance(gamma, list):
            gamma = numpy.array(gamma)

        eta = numpy.array([])
        x_min = 0
        for idx, x_max in enumerate(self.params['x_max']):
            param_temp = {}
            # gamma filter
            gamma_filter = numpy.logical_and(x_min < gamma, gamma <= x_max)
            gamma_s = gamma[gamma_filter]
            if len(gamma_s) > 0:
                # get all parameters in row except x_max
                for key, value in self.params.items():
                    if key not in ['x_max']:
                        param_temp[key] = value[idx]
                # append eta values values
                if self.kind == 'Herschel-Bulkley':
                    eta[gamma_filter] = get_eta_herschel_bulkley(
                        gamma_s, **param_temp)
                elif self.kind == 'Newton':
                    eta[gamma_filter] = self.params['eta'][idx]
            x_min = 1. * x_max

        return eta

    def get_tau(self, gamma: float|list|numpy.ndarray):
        """
        Returns shear stress for given shear rate(s) depending on dynamic viscosity model
        
        Parameters
        ----------
        gamma : float | numpy.ndarray | list
            Shear rate(s)

        Returns
        -------
        numpy.ndarray
        """
        return numpy.multiply(self.get_eta(gamma), gamma)

    def get_tau_gamma(self, gamma: float | list | numpy.ndarray):
        """
        Returns equivalent of mass specific power for given shear rate(s) depending on dynamic viscosity model

        Parameters
        ----------
        gamma : float | numpy.ndarray | list
            Shear rate(s)

        Returns
        -------
        numpy.ndarray
        """
        return numpy.multiply(self.get_eta(gamma), numpy.power(gamma, 2))

    def fit_parameters(self, df: pandas.DataFrame, gamma_max=None, log_fit=True):
        """
        Gets a DataFrame and returns a viscosity object with parameters fitted to data
        and split in ranges according to gamma_max list (if given)
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with viscosity data
        gamma_max : None or list, optional
            if list then the fits are split according the gamma_max ranges
        log_fit: Bool
            fits to log10 of y_values if True. linear values fit if False

        Returns
        -------
        Viscosity
        """
        # copy object
        new_visc = self.copy()
        new_visc.params = {}
        if gamma_max is None:
            gamma_max = [numpy.inf]
        # split DataFrame by gamma_max
        dfs = split_df_by_bounds(df, gamma_max)

        for idx, limit in enumerate(gamma_max):
            fit_df = dfs[idx].copy()

            if new_visc.kind == 'Herschel-Bulkley':
                fit_params = get_herschel_bulkley_fit(fit_df, log_fit)

                for param in fit_params:
                    if param in new_visc.params:
                        new_visc.params[param] += [fit_params[param]]
                    else:
                        new_visc.params[param] = [fit_params[param]]
        new_visc.params['x_max'] = copy.deepcopy(gamma_max)

        return new_visc

    def plot_on_ax(self, ax, gamma: float | list | numpy.ndarray, y: str = 'eta', **kwargs):
        """
        Plots viscosity data on given ax

        Parameters
        ----------
        ax : Axes
            Pyplot.Ax object for the data plot
        gamma : float | numpy.ndarray | list
            Shear rate(s)
        y : str
            name of quantity.
                'eta' : dynamic viscosity in Pas (default)
                'tau' : shear stress in Pa
                'tau_gamma' : product of shear stress and shear rate in Pa/s
        kwargs

        Returns
        -------
        None
        """
        if y == 'eta':
            y_plot = self.get_eta(gamma)
        elif y == 'tau':
            y_plot = self.get_tau(gamma)
        elif y == 'tau_gamma':
            y_plot = self.get_tau_gamma(gamma)
        else:
            y_plot = gamma
        io_node.plot_on_ax(x_plot=gamma, y_plot=y_plot, ax=ax, c=kwargs.setdefault('c', kit_colors.black),
                           lw=kwargs.setdefault('lw', 1), linestyle=kwargs.setdefault('linestyle', '-'),
                           marker=kwargs.setdefault('marker', 'o'), mfc=kwargs.setdefault('mfc', kwargs['c']),
                           mec=kwargs.setdefault('mec', kwargs['c']), ms=kwargs.setdefault('ms', 6),
                           mew=kwargs.setdefault('mew', 1), alpha=kwargs.setdefault('alpha', 0.9),
                           label=kwargs.setdefault('label', self.name))  # label
