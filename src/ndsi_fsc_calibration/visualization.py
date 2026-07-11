from typing import Tuple

import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ndsi_fsc_calibration.fit_linear_model import compute_correlation_coefficient_from_weights, fit_regression


def salomonson_appel(ndsi):
    """ "'FRA6T' or 'universal' equation for NDSI calculation.
    See Salomonson, Vincent V., and Igor Appel. "Development of the Aqua MODIS NDSI fractional snow cover algorithm and
      validation results." IEEE Transactions on geoscience and remote sensing 44.7 (2006): 1747-1756."""
    return 1.45 * ndsi - 0.01


def scatter_plot_with_fit(
    data: xr.DataArray,
    eval_prod_name: str,
    fig: Figure,
    ax: Axes,
    quantile_min: float = 0.2,
    quantile_max: float = 0.9,
    fsc_min: int = 0,
    fsc_max: int = 100,
) -> Tuple[Figure, Axes]:
    """Generate a scatter plot of NDSI, FSC correspondences and display a linear fit together with Salomonson and Appel fit.

    Args:
        data (xr.DataArray): array of the (NDSI, FSC) correspondences generated using match.Scatter
        eval_prod_name (str): evaluation product identifier (ex. VNP10A1) to print in the plot
        fig (Figure): a matplolib figure
        # ax (Axes): a matplolib axes
        quantile_min (float, optional): minimum quantile to normalize the colramp for enhanced visualization. Defaults to 0.2.
        quantile_max (float, optional): maximum quantile to normalize the colramp for enhanced visualization. Defaults to 0.9.

    Returns:
        Tuple[Figure, Axes]: the output plotting objects for further work or export
    """
    data_to_plt = data.transpose("fsc", "ndsi")
    data_to_plt = data_to_plt.sel(fsc=slice(fsc_min, fsc_max))
    data_to_plt = data_to_plt.assign_coords({"fsc": data_to_plt.coords["fsc"] / 100, "ndsi": data_to_plt.coords["ndsi"] / 100})
    coeff_slope_ndsi, intercept_ndsi, score = fit_regression(data_to_plt)
    # Invert model to draw regression
    coeff_slope = 1 / coeff_slope_ndsi
    intercept = -intercept_ndsi
    distr_min, distr_max = np.quantile(data_to_plt, quantile_min), np.quantile(data_to_plt, quantile_max)

    # Create colormap
    cmap = plt.cm.bone.copy()
    cmap.set_under("black")  # for values < 1
    cmap.set_bad("black")
    # Normalization: only 1–100 use the colormap
    norm = colors.LogNorm(vmin=distr_min if distr_min > 0 else 1, vmax=distr_max, clip=False)
    xx, yy = np.meshgrid(data_to_plt.coords["ndsi"], data_to_plt["fsc"])
    scatter = ax.pcolormesh(
        xx,
        yy,
        data_to_plt,
        norm=norm,
        cmap=cmap,
    )

    regression_x_axis = np.arange(0, 1, 0.01)
    # pearson_corr_coeff = compute_correlation_coefficient_from_weights(data_to_plt)

    ax.plot(
        regression_x_axis,
        regression_x_axis * coeff_slope + intercept,
        ":",
        lw=1.5,
        color="chocolate",
        label=f"Linear fit slope={float(coeff_slope):.2f},intercept={float(intercept):.2f}, R²={score:.2f} ",
    )
    # xax = data_to_plt.coords["fsc"].values / 100
    ax.plot(
        regression_x_axis,
        salomonson_appel(regression_x_axis),
        color="chocolate",
        linewidth=1.5,
        label="(Salomonson and Appel, 2006)",
    )

    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), draggable=True, fontsize=12)

    ax.set_ylabel("S2 FSC")
    ax.set_xlabel(f"{eval_prod_name} NDSI")
    # ax.set_yticks(np.arange(np.ceil(fsc_min / 10) * 10, 100, 10))
    ax.set_ylim(fsc_min / 100, fsc_max / 100)
    ax.set_xlim(0, 1)
    ax.grid(True)
    # fig.colorbar(scatter)
    cbar = fig.colorbar(scatter, extend="max")
    cbar_ticks = np.array([1e1, 1e2])
    cbar_labels = [f"{tick:n}" for tick in cbar_ticks]
    cbar.set_ticks(cbar_ticks, labels=cbar_labels)
    cbar.ax.set_ylabel("# of matches", rotation=270)
    return fig, ax
