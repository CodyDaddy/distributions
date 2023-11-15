import numpy as np
from scipy.optimize import curve_fit


def get_eta_herschel_bulkley(gamma, tau_0: float, k: float, n: float):
    return tau_0 / gamma + k * np.power(gamma, n - 1)


def get_eta_herschel_bulkley_log(gamma, tau_0: float, k: float, n: float):
    """
    only because curve_fit doesn't pass args to callable :(
    Returns
    -------

    """
    return np.log10(tau_0 / gamma + k * np.power(gamma, n - 1))


def get_herschel_bulkley_fit(df, log_fit):
    gamma = df['shear rate'].to_numpy()
    eta = df['dynamic viscosity'].to_numpy()
    if log_fit:
        eta = np.log10(eta)
        popt, pcov = curve_fit(get_eta_herschel_bulkley_log, gamma, eta)
    else:
        popt, pcov = curve_fit(get_eta_herschel_bulkley, gamma, eta)
    return {'tau_0': popt[0], 'k': popt[1], 'n': popt[2]}


