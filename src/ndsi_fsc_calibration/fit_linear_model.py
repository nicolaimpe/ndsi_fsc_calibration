from typing import Tuple

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression


def compute_correlation_coefficient_from_weights(weights: xr.DataArray) -> float:
    """Compute Pearson correlation coefficient for a discrete (X,Y) dataset of the number of occurrences for each x,y pair.

    Args:
        weights (xr.DataArray): array with X,Y as coordinates and number of occurrences as values

    Returns:
        float: the Pearson correlation coefficient
    """
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


def fit_regression(data_to_fit: xr.DataArray) -> Tuple[float, float, float]:
    """Fit a linear regression model for a discrete (X,Y) dataset of the number of occurrences for each x,y pair.

    Args:
        data_to_fit (xr.DataArray): array with X,Y as coordinates and number of occurrences as values

    Returns:
        Tuple[float, float, float]: linear model slope, intercept and R**2 score
    """
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
