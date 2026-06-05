from scipy.optimize import curve_fit

from .statistics import chi2, aic, aicc, bic
from .models.factory import set_model_parameters


def make_pred_function(gamma_model, energies, kind):

    def pred(_, *pars):
        set_model_parameters(gamma_model, kind, pars)

        return (
            gamma_model(energies)
            * energies**2
        ).value

    return pred


def fit_model(
    kind,
    gamma_model,
    energy,
    flux,
    sigma,
    p0,
):

    pred = make_pred_function(
        gamma_model,
        energy,
        kind,
    )

    popt, cov = curve_fit(
        pred,
        energy.value,
        flux.value,
        sigma=sigma.value,
        p0=p0,
        maxfev=20000,
    )

    set_model_parameters(
        gamma_model,
        kind,
        popt,
    )

    #ndof = len(energy) - len(popt)

    #chi2_val = chi2(
    #    gamma_model,
    #    energy,
    #    flux,
    #    sigma,
    #) / ndof

    chi2_total = chi2(
        gamma_model,
        energy,
        flux,
        sigma,
    )

    k = len(popt)
    n = len(energy)

    chi2_red = chi2_total / (n - k)

    aic_value = aic(
        chi2_total.value,
        k,
    )

    aicc_value = aicc(
        chi2_total.value,
        k,
        n,
    )

    bic_value = bic(
        chi2_total.value,
        k,
        n,
    )
    
    
    return {
        "parameters": popt,
        "chi2": chi2_total.value,
        "chi2_red": chi2_red.value,
        "aic": aic_value,
        "aicc": aicc_value,
        "bic": bic_value,
    }
