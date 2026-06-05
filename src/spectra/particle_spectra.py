import astropy.units as u
from naima.models import (
    PowerLaw,
    ExponentialCutoffPowerLaw,
    BrokenPowerLaw,
    LogParabola
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

