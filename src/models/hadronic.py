import astropy.units as u

from gammapy.modeling.models import NaimaSpectralModel
from naima.models import PionDecay

from ..spectra.particle_spectra import make_parent_spectrum


def make_hadronic_model(
    kind,
    distance,
    nh=1.0*u.cm**-3
):
    """
    Create a gamma-ray model based on pion decay.
    """

    parent = make_parent_spectrum(
        kind,
        hadronic=True
    )

    return NaimaSpectralModel(
        PionDecay(parent, nh=nh),
        distance=distance
    )
