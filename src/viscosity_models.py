import numpy
from scipy.optimize import curve_fit


def get_eta_herschel_bulkley(gamma: float | numpy.ndarray | list, tau_0: float, k: float, n: float):
    """
    Returns dynamic viscosity according to Herschel-Bulkley model
    
    Parameters
    ----------
    gamma : float | numpy.ndarray | list
        Shear rate(s)
    tau_0 : float
        yield shear stress 
    k : float
        consistency k 
    n : float
        flow index

    Returns
    -------
    numpy.ndarray
    """
    return tau_0 / gamma + k * numpy.power(gamma, n - 1)


def get_eta_herschel_bulkley_log(gamma: float | numpy.ndarray | list, tau_0: float, k: float, n: float):
    """
    Returns log10 of dynamic viscosity according to Herschel-Bulkley model
    only because curve_fit doesn't pass args to callable :(

    Parameters
    ----------
    gamma : float | numpy.ndarray | list
        Shear rate(s)
    tau_0 : float
        yield shear stress
    k : float
        consistency k
    n : float
        flow index

    Returns
    -------
    numpy.ndarray
    """
    return numpy.log10(get_eta_herschel_bulkley(gamma=gamma, tau_0=tau_0, k=k, n=n))


def get_herschel_bulkley_fit(gamma: numpy.ndarray | list, eta: numpy.ndarray | list, log_fit: bool = False):
    """
    Fits parameters for Herschel-Bulkley model from viscosity measurement data

    Parameters
    ----------
    gamma : numpy.ndarray | list
        shear rate data
    eta : numpy.ndarray | list
        dynamic viscosity at given shear rates
    log_fit : bool, optional
        False : calculates fit error from absolute values, default

        True : calculates fit error from log10 values

    Returns
    -------
    dict
        Dictionary with parameters
    """
    if isinstance(gamma, list):
        gamma = numpy.array(gamma)
    if isinstance(eta, list):
        eta = numpy.array(eta)
    if log_fit:
        eta = numpy.log10(eta)
        popt, pcov = curve_fit(get_eta_herschel_bulkley_log, gamma, eta)
    else:
        popt, pcov = curve_fit(get_eta_herschel_bulkley, gamma, eta)
    return {'tau_0': popt[0], 'k': popt[1], 'n': popt[2]}


