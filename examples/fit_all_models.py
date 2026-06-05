import astropy.units as u
import pandas as pd

from src.io import load_flux_points

from src.fitting import fit_model

from src.plotting import plot_all_models

from src.models.factory import (
    make_gamma_models,
    initial_guess,
)



# ============================================================
# CONFIGURATION
# ============================================================

FP_FILE = "data/HWC_J1908_06.txt"

DISTANCE = 2.37 * u.kpc

SPECTRAL_KINDS = [
    "PL",
    "ECPL",
    "BPL",
    "LogParabola",
]

# ============================================================
# MAIN
# ============================================================
def main():

    energy, flux, flux_err, flux_err_plus, flux_err_minus = (
        load_flux_points(FP_FILE)
    )

    results = []


    # --------------------------------------------------------
    # Fit loop
    # --------------------------------------------------------

    for kind in SPECTRAL_KINDS:

        print(f"\nTesting {kind}")

        gamma_PD, gamma_IC = make_gamma_models(
            kind,
            DISTANCE,
        )

        p0_PD, p0_IC = initial_guess(kind)

        result_PD = fit_model(
            kind=kind,
            gamma_model=gamma_PD,
            energy=energy,
            flux=flux,
            sigma=flux_err,
            p0=p0_PD,
        )

        result_IC = fit_model(
            kind=kind,
            gamma_model=gamma_IC,
            energy=energy,
            flux=flux,
            sigma=flux_err,
            p0=p0_IC,
        )

    results.append(
        {
            "kind": kind,

            "popt_PD": result_PD["parameters"],
            "chi2_PD": result_PD["chi2_red"],
            "aic_PD": result_PD["aic"],
            "aicc_PD": result_PD["aicc"],
            "bic_PD": result_PD["bic"],

            "popt_IC": result_IC["parameters"],
            "chi2_IC": result_IC["chi2_red"],
            "aic_IC": result_IC["aic"],
            "aicc_IC": result_IC["aicc"],
            "bic_IC": result_IC["bic"],
        }
    )


    # --------------------------------------------------------
    # Best models
    # --------------------------------------------------------

    best_PD = min(
        results,
        key=lambda r: r["aicc_PD"]
    )

    best_IC = min(
        results,
        key=lambda r: r["aicc_IC"]
    )

    print(
        f"\nBest PD model: "
        f"{best_PD['kind']} "
        f"(AIC-corrected = {best_PD['aicc_PD']:.2f})"
    )

    print(
        f"Best IC model: "
        f"{best_IC['kind']} "
        f"(AIC-corrected = {best_IC['aicc_IC']:.2f})"
    )


    df = pd.DataFrame(
        [
            {
                "Model": r["kind"],
                "Chi2_PD": r["chi2_PD"],
                "AICc_PD": r["aicc_PD"],
                "Chi2_IC": r["chi2_IC"],
                "AICc_IC": r["aicc_IC"],
            }
            for r in results
        ]
    )

    print(df.to_string(index=False))
    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_all_models(
        results=results,
        energy=energy,
        flux=flux,
        flux_err_plus=flux_err_plus,
        flux_err_minus=flux_err_minus,
        distance=DISTANCE,
    )

if __name__ == "__main__":
    main()
