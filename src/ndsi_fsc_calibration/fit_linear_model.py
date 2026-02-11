import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression


def compute_correlation_coefficient_from_weights(weights: xr.DataArray) -> float:
    i = np.arange(weights.sizes["ndsi"])  # X values
    j = np.arange(weights.sizes["fsc"])  # Y values

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
    xx, yy = np.meshgrid(data_to_fit.coords["ndsi"].values, data_to_fit.coords["fsc"].values)
    model_x_data = xx.reshape((-1, 1))
    model_y_data = yy.reshape((-1, 1))
    weights = data_to_fit.values.ravel()
    regression = LinearRegression().fit(X=model_y_data, y=model_x_data, sample_weight=weights.T)
    return (
        regression.coef_[0][0],
        regression.intercept_[0],
        regression.score(model_y_data, model_x_data, weights),
    )
