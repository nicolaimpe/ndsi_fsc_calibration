import numpy as np
import xarray as xr
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression


def compute_correlation_coefficient_from_weights(weights: xr.DataArray) -> float:
    i = np.arange(weights.sizes["y"])  # X values
    j = np.arange(weights.sizes["x"])  # Y values

    w = weights.values

    # Compute expectations
    N = np.sum(w)
    sum_x = np.sum(i[:, None] * w)
    sum_y = np.sum(j[None, :] * w)
    sum_x_square = np.sum((i[:, None] ** 2) * w)
    sum_y_square = np.sum((j[None, :] ** 2) * w)
    sum_x_y = np.sum((i[:, None] * j[None, :]) * w)

    # Pearson correlation
    r = (N * sum_x_y - sum_x * sum_y) / np.sqrt((N * sum_x_square - (sum_x) ** 2) * (N * sum_y_square - (sum_y) ** 2))
    return r


def fit_regression(data_to_fit: xr.DataArray):
    xx, yy = np.meshgrid(data_to_fit.coords["x"].values, data_to_fit.coords["y"].values)
    model_x_data = xx.reshape((-1, 1))
    model_y_data = yy.reshape((-1, 1))
    weights = data_to_fit.values.ravel()
    regression = LinearRegression().fit(X=model_y_data, y=model_x_data, sample_weight=weights.T)
    return (
        regression.coef_[0][0],
        regression.intercept_[0],
        regression.score(model_y_data, model_x_data, weights),
    )


def salomonson_appel(ndsi):
    return 1.45 * ndsi - 0.01


def fancy_scatter_plot_with_fit(data_to_plt: xr.DataArray, ax: Axes, perc_min: float = 0.2, perc_max: float = 0.9):
    data_to_plt = data_to_plt.transpose("y", "x")

    coeff_slope_ndsi, intercept_ndsi, score = fit_regression(data_to_plt)
    # Invert model to draw regression
    coeff_slope = 1 / coeff_slope_ndsi
    intercept = -intercept_ndsi
    distr_min, distr_max = np.quantile(data_to_plt, perc_min), np.quantile(data_to_plt, perc_max)
    ax.pcolormesh(
        data_to_plt.coords["x"].values,
        data_to_plt.coords["y"].values,
        data_to_plt,
        norm=colors.LogNorm(vmin=distr_min if distr_min > 0 else 1, vmax=distr_max, clip=True),
        cmap=cm.bone,
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
    xax = data_to_plt.coords["x"].values
    ax.plot(xax, salomonson_appel(xax), color="chocolate", linewidth=1.5, label="(Salomonson and Appel, 2006)")

    ax.grid(False)
    ax.legend(loc="lower center", draggable=True)

    ax.set_ylabel("S2 FSC [%]")
    ax.set_xlabel("VIIRS NDSI [%]")
