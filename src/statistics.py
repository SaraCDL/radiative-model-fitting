import numpy as np
import astropy.units as u

def chi2(model, energy, flux, flux_err):
    """
    Chi-square using flux points already given as E² dN/dE.
    """

    model_sed = (model(energy) * energy**2).to("erg cm-2 s-1")

    data_sed = flux.to("erg cm-2 s-1")

    data_err = flux_err.to("erg cm-2 s-1")

    return ((model_sed - data_sed)**2 / data_err**2).sum()

def aic(chi2, k):
    return chi2 + 2 * k

def aicc(chi2, k, n):
    return chi2 + 2*k + (2*k*(k+1))/(n-k-1)

def bic(chi2, k, n):
    return chi2 + k * np.log(n)
