import astropy.units as u

from gammapy.modeling.models import NaimaSpectralModel

from naima.models import (
    PowerLaw,
    ExponentialCutoffPowerLaw,
    BrokenPowerLaw,
    LogParabola,
    InverseCompton,
    PionDecay,
)

def make_parent_spectrum(kind, hadronic=True):

    A = 1e35 / u.eV if hadronic else 1e33 / u.eV

    if kind == "PL":
        return PowerLaw(
            amplitude=A,
            e_0=10 * u.TeV,
            alpha=2.2 if hadronic else 3.0
        )

    if kind == "ECPL":
        return ExponentialCutoffPowerLaw(
            amplitude=A,
            e_0=10 * u.TeV,
            alpha=2.2 if hadronic else 3.0,
            e_cutoff=300 * u.TeV,
            beta=1
        )

    if kind == "BPL":
        return BrokenPowerLaw(
            amplitude=A,
            e_0=10 * u.TeV,
            alpha_1=2.0,
            alpha_2=3.0,
            e_break=30 * u.TeV
        )

    if kind == "LogParabola":
        return LogParabola(
            amplitude=A,
            e_0=10 * u.TeV,
            alpha=2.2 if hadronic else 3.0,
            beta=0.2
        )


def make_gamma_models(kind, distance):
    """
    Create both hadronic and leptonic gamma-ray models
    for a given parent particle spectrum.
    """
    nh=1.0*u.cm**-3
    seed_photon_fields="CMB"

    parent_PD = make_parent_spectrum(kind, True)
    parent_IC = make_parent_spectrum(kind, False)

    gamma_PD = NaimaSpectralModel(PionDecay(parent_PD, nh=nh, distance=distance))
    gamma_IC = NaimaSpectralModel(InverseCompton(parent_IC, seed_photon_fields=seed_photon_fields), distance=distance)

    return gamma_PD, gamma_IC

def set_model_parameters(model, kind, pars):

    if kind == "PL":
        names = ["amplitude", "alpha"]

    elif kind == "ECPL":
        names = ["amplitude", "alpha", "e_cutoff"]

    elif kind == "BPL":
        names = ["amplitude", "alpha_1", "alpha_2", "e_break"]

    elif kind == "LogParabola":
        names = ["amplitude", "alpha", "beta"]

    for n, v in zip(names, pars):
        model.parameters[n].value = v
        
def initial_guess(kind):

    if kind == "PL":
        return [1e35, 2.2], [1e33, 3.0]

    if kind == "ECPL":
        return [1e35, 2.2, 300], [1e33, 3.0, 100]

    if kind == "BPL":
        return [1e35, 2.0, 3.0, 30], [1e33, 2.0, 3.0, 20]

    if kind == "LogParabola":
        return [1e35, 2.2, 0.2], [1e33, 3.0, 0.2]
        

