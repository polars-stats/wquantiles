"""
Library to compute weighted quantiles, including the weighted median, of
numpy arrays.
"""

import polars as pl
from polars import DataFrame, Series
import jax.numpy as np


__version__ = "0.4"


def quantile_1d(frame: DataFrame, column: str, weights: str, quantile: float):
    """
    Compute the weighted quantile of a 1D numpy array.

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile_1d : float
        The output value.
    """
    assert 0. <= quantile <= 1., (
        "quantile must have a value between 0. and 1."
    )
    # Sort the data
    # ind_sorted = np.argsort(data)
    # sorted_data = data[ind_sorted]
    # sorted_weights = weights[ind_sorted]
    # TODO: Check that the weights do not sum zero
    # assert Sn != 0, "The sum of the weights must not be zero"
    # Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
    return (
        frame
        .sort(column)
        .with_columns(
            (pl.col(weights).cumsum() - 0.5 * pl.col(weights))
                / pl.col(weights).cumsum()
        )
        .select(pl.col(column) * pl.col(weights))
        .interpolate()
        .quantile(quantile)
    )
    # Get the value of the weighted median
    # return np.interp(quantile, Pn, sorted_data)


def quantile(data: DataFrame | Series, weights: str, quantile: float):
    """
    Weighted quantile of an array with respect to the last axis.

    Parameters
    ----------
    data : ndarray
        Input array.
    weights : ndarray
        Array with the weights. It must have the same size of the last 
        axis of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile : float
        The output value.
    """
    if isinstance(data, Series):
        return quantile_1d(data, weights, quantile)
    elif isinstance(data, DataFrame):
        n = data.shape
        imr = data.reshape((np.prod(n[:-1]), n[-1]))
        result = np.apply_along_axis(quantile_1d, -1, imr, weights, quantile)
        return result.reshape(n[:-1])


def median(data: DataFrame | Series, weights: str):
    """
    Weighted median of an array with respect to the last axis.

    Alias for `quantile(data, weights, 0.5)`.
    """
    return quantile(data, weights, 0.5)
