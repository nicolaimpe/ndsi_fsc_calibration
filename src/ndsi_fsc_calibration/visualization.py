from typing import Tuple

import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ndsi_fsc_calibration.fit_linear_model import compute_correlation_coefficient_from_weights, fit_regression


def salomonson_appel(ndsi):
    return 1.45 * ndsi - 0.01


def scatter_plot_with_fit(
    data_to_plt: xr.DataArray,
    eval_prod_name: str,
    fig: Figure,
    ax: Axes,
    quantile_min: float = 0.2,
    quantile_max: float = 0.9,
) -> Tuple[Figure, Axes]:
    data_to_plt = data_to_plt.transpose("fsc", "ndsi")

    coeff_slope_ndsi, intercept_ndsi, score = fit_regression(data_to_plt)
    # Invert model to draw regression
    coeff_slope = 1 / coeff_slope_ndsi
    intercept = -intercept_ndsi
    distr_min, distr_max = np.quantile(data_to_plt, quantile_min), np.quantile(data_to_plt, quantile_max)

    # Create colormap
    cmap = plt.cm.Blues_r.copy()
    cmap.set_under("white")  # for values < 1
    cmap.set_bad("white")
    # Normalization: only 1–100 use the colormap
    norm = colors.LogNorm(vmin=distr_min if distr_min > 0 else 1, vmax=distr_max, clip=False)
    scatter = ax.imshow(
        data_to_plt,
        norm=norm,
        cmap=cmap,
    )

    regression_x_axis = np.arange(0, 100)
    pearson_corr_coeff = compute_correlation_coefficient_from_weights(data_to_plt)

    ax.plot(
        regression_x_axis,
        regression_x_axis * coeff_slope + intercept,
        ":",
        lw=1.5,
        color="chocolate",
        label=f"Linear fit slope={float(coeff_slope):.2f},intercept={float(intercept):.2f}, R²={score:.2f}, r={pearson_corr_coeff:.2f} ",
    )
    xax = data_to_plt.coords["fsc"].values
    ax.plot(xax, salomonson_appel(xax), color="chocolate", linewidth=1.5, label="(Salomonson and Appel, 2006)")

    ax.grid(False)
    ax.legend(loc="lower center", draggable=True)

    ax.set_ylabel("S2 FSC [%]")
    ax.set_xlabel(f"{eval_prod_name} NDSI [%]")
    ax.set_ylim(10, 95)
    ax.set_xlim(0, 100)
    ax.grid(True)
    plt.colorbar(scatter)
    return fig, ax
