
import pickle
from scipy import signal
from cgi import test
import os
import torch
import random
import getpass
import pickle as pkl
import numpy as np
import itertools


list_of_features = ["abs_energy", "absolute_maximum", "count_above_mean", "first_location_of_maximum", "last_location_of_maximum", "longest_strike_above_mean", "mean_change", "variation_coefficient"]

def applyAndReturnAllFeatures(x):
    """
    Applies all features in the list of features and returns a dictionary with the feature names as keys and the feature values as values.

    :param x: the time series to calculate the features of
    :type x: numpy.ndarray
    :return: the dictionary with the feature names as keys and the feature values as values
    :return type: dict
    """
    return {f: globals()[f](x) for f in list_of_features}
    #return [globals()[f](x) for f in list_of_features]


def abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    #if not isinstance(x, (np.ndarray, pd.Series)):
    #    x = np.asarray(x)
    return np.dot(x, x)

def absolute_maximum(x):
    """
    Calculates the highest absolute value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.max(np.absolute(x)) if len(x) > 0 else np.NaN

def count_above_mean(x):
    """
    Returns the number of values in x that are higher than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    m = np.mean(x)
    return np.where(x > m)[0].size

def first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    #if not isinstance(x, (np.ndarray, pd.Series)):
    #    x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN

def last_location_of_maximum(x):
    """
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def longest_strike_above_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    #if not isinstance(x, (np.ndarray, pd.Series)):
    #    x = np.asarray(x)
    return np.max(_get_length_sequences_where(x > np.mean(x))) if x.size > 0 else 0

def mean_change(x):
    """
    Average over time series differences.

    Returns the mean over the differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1}  x_{i+1} - x_{i} = \\frac{1}{n-1} (x_{n} - x_{1})

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN

def variation_coefficient(x):
    """
    Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan

def _get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.

    Examples
    --------
    >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    :param x: An iterable containing only 1, True, 0 and False values
    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]