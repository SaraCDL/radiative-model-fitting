import matplotlib.pyplot as plt
import astropy.units as u

from .models.factory import (
    make_gamma_models,
    set_model_parameters,
)

import os

os.makedirs("plots", exist_ok=True)


def plot_all_models(
    results,
    energy,
    flux,
    flux_err_plus,
    flux_err_minus,
    distance,
    source_name="HWC_J1908_063",
):
    """
    Plot all fitted hadronic and leptonic gamma-ray models.
    """

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(
        figsize=(5.0, 3.8),
        dpi=200,
    )

    colors = {
        "PL": "tab:purple",
        "ECPL": "tab:red",
        "BPL": "tab:green",
        "LogParabola": "tab:blue",
    }

    for result in results:

        kind = result["kind"]

        gamma_PD, gamma_IC = make_gamma_models(
            kind,
            distance,
        )

        # Hadronic best fit
        set_model_parameters(
            gamma_PD,
            kind,
            result["popt_PD"],
        )

        # Leptonic best fit
        set_model_parameters(
            gamma_IC,
            kind,
            result["popt_IC"],
        )

        # Hadronic
        gamma_PD.plot(
            [0.1, 300] * u.TeV,
            energy_power=2,
            ax=ax,
            color=colors[kind],
            ls="-",
            label=(
                f"PD {kind} "
                f"(χ²={result['chi2_PD']:.2f})"
            ),
        )

        # Leptonic
        gamma_IC.plot(
            [0.1, 300] * u.TeV,
            energy_power=2,
            ax=ax,
            color=colors[kind],
            ls="--",
            label=(
                f"IC {kind} "
                f"(χ²={result['chi2_IC']:.2f})"
            ),
        )

    # Flux points
    ax.errorbar(
        energy.value,
        flux.value,
        yerr=[
            flux_err_minus.value,
            flux_err_plus.value,
        ],
        fmt="o",
        color="black",
        markersize=5,
        capsize=3,
        label="HAWC data",
        zorder=100,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1, 300)
    ax.set_ylim(1e-14, None)

    ax.set_xlabel(
        r"$E\,[\mathrm{TeV}]$"
    )

    ax.set_ylabel(
        r"$E^2 dN/dE\,[\mathrm{TeV\,cm^{-2}\,s^{-1}}]$"
    )

    ax.legend(
        fontsize=6,
        ncol=2,
    )

    plt.tight_layout()

    plt.savefig(
        f"plots/{source_name}_all_models.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        f"plots/{source_name}_all_models.pdf",
        bbox_inches="tight",
    )

    plt.show()
