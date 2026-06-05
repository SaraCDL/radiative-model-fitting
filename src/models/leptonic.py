import astropy.units as u

from gammapy.modeling.models import NaimaSpectralModel
from naima.models import InverseCompton

from ..spectra.particle_spectra import make_parent_spectrum


def make_leptonic_model(
    kind,
    distance,
    seed_photon_fields="CMB"
):
    """
    Create a gamma-ray model based on inverse Compton.
    """

    parent = make_parent_spectrum(
        kind,
        hadronic=False
    )

    return NaimaSpectralModel(
        InverseCompton(
            parent,
            seed_photon_fields=seed_photon_fields
        ),
        distance=distance
    )
