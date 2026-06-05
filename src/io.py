import numpy as np
import pandas as pd
import astropy.units as u

from astropy.table import Table
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import LogParabolaSpectralModel

def load_flux_points(filename):

    data = np.loadtxt(filename)

    energy = data[:,0] * u.TeV

    flux = data[:,1] * u.Unit("TeV cm-2 s-1")

    flux_err_plus = data[:,2] * u.Unit("TeV cm-2 s-1")

    flux_err_minus = data[:,3] * u.Unit("TeV cm-2 s-1")

    flux_err = 0.5 * (flux_err_plus + flux_err_minus)

    return (
        energy,
        flux,
        flux_err,
        flux_err_plus,
        flux_err_minus,
    )
