import os
from random import sample
import tensorflow as tf
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd


# util

def mfcc(stfts):
    sample_rate = 1024
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]# .value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 10.0, 250.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]

    return mfccs

## Time domain features

def power(x):
    # Calculate power over each window [J/s]
    return tf.math.square(tf.abs(x))/x.size


def cross_correlation(x, y):
    # Calculate cross-correlation between two signals
    return tf.signal.fftshift(tf.signal.correlate(x, y, fft_length=x.size, pad_end=False))

def autocorrelation(x):
    # Calculate autocorrelation over each window
    return tf.signal.fftshift(tf.signal.correlate(x, x, fft_length=x.size, pad_end=False))

def cross_correlation_matrix(x, y):
    # Calculate cross-correlation matrix between two signals with 2D FFT
    return tf.signal.fftshift(tf.signal.correlate2d(x, y, fft_length=x.shape[1], pad_end=False))




## Freq domain features -- complex statistics

def fft_aggregation(x,axis=1): #TODO: Debug for other statistics
    # Aggregate FFT result over each window to find FFT means
    return tf.math.reduce_mean(tf.math.square(tf.abs(x)), axis=axis)

def autocorrelation(x, lag,len_x):
    """
    Calculates the autocorrelation of the specified lag, according to the formula [1]

    .. math::

        \\frac{1}{(n-l)\\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`n` is the length of the time series :math:`X_i`, :math:`\\sigma^2` its variance and :math:`\\mu` its
    mean. `l` denotes the lag.

    .. rubric:: References

    [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    # This is important: If a series is passed, the product below is calculated
    # based on the index, which corresponds to squaring the series.
    # if isinstance(x, pd.Series):
    #     x = x.values
    # if len(x) < lag:
    #     return np.nan
    # Slice the relevant subseries based on the lag
    y1 = x[: (len_x - lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(x)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(x)
    if np.isclose(v, 0):
        return np.NaN
    else:
        return sum_product / ((len(x) - lag) * v)

def agg_autocorrelation(x, param):
    """
    Descriptive statistics on the autocorrelation of the time series.

    Calculates the value of an aggregation function :math:`f_{agg}` (e.g. the variance or the mean) over the
    autocorrelation :math:`R(l)` for different lags. The autocorrelation :math:`R(l)` for lag :math:`l` is defined as

    .. math::

        R(l) = \\frac{1}{(n-l)\\sigma^2} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`X_i` are the values of the time series, :math:`n` its length. Finally, :math:`\\sigma^2` and
    :math:`\\mu` are estimators for its variance and mean
    (See `Estimation of the Autocorrelation function <http://en.wikipedia.org/wiki/Autocorrelation#Estimation>`_).

    The :math:`R(l)` for different lags :math:`l` form a vector. This feature calculator applies the aggregation
    function :math:`f_{agg}` to this vector and returns

    .. math::

        f_{agg} \\left( R(1), \\ldots, R(m)\\right) \\quad \\text{for} \\quad m = max(n, maxlag).

    Here :math:`maxlag` is the second parameter passed to this function.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"f_agg": x, "maxlag", n} with x str, the name of a numpy function
                  (e.g. "mean", "var", "std", "median"), its the name of the aggregator function that is applied to the
                  autocorrelations. Further, n is an int and the maximal number of lags to consider.
    :type param: list
    :return: the value of this feature
    :return type: float
    """
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    THRESHOLD_TO_USE_FFT = 1250
    var = np.var(x)
    n = len(x)
    max_maxlag = max([config["maxlag"] for config in param])

    if np.abs(var) < 10 ** -10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, adjusted=True, fft=n > THRESHOLD_TO_USE_FFT, nlags=max_maxlag)[1:]
    return [
        (
            'f_agg_"{}"__maxlag_{}'.format(config["f_agg"], config["maxlag"]),
            getattr(np, config["f_agg"])(a[: int(config["maxlag"])]),
        )
        for config in param
    ]
