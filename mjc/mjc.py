from math import pow, inf
from statistics import stdev
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, grid, show
import numpy as np


def minimumMJC(s1, s2, dXYlimit=inf, beta=1, showPlot=False, std_s1=None, std_s2=None, tavg_s1=None, tavg_s2=None,
               overrideChecks=False):
    dXY_a, abandoned_a, std_s1, std_s2, tavg_s1, tavg_s2, s1, s2 = MJC(s1, s2, dXYlimit, beta, showPlot, std_s1, std_s2,
                                                                       tavg_s1, tavg_s2, return_args=True,
                                                                       override_checks=overrideChecks)

    dXY_b, abandoned_b = MJC(s2, s1, dXYlimit, beta, showPlot, std_s2, std_s1, tavg_s2, tavg_s1, override_checks=True)

    return min(dXY_a, dXY_b), abandoned_a and abandoned_b


def MJC(s1, s2, dxy_limit=inf, beta=1, show_plot=False, std_s1=None, std_s2=None, tavg_s1=None, tavg_s2=None,
        return_args=False, override_checks=False):
    """
    Minimum Jump Cost (MJC) dissimiliarity algorithm.
    This algorithm implements the MJC algorithm devised by Joan Serra and Josep Lluis Arcos (2012). This algorithm was
    shown to outperform the Dynamic Time Warp (DTW) dissimilarity algorithm on several datasets.
    
    This function takes two time series s1 and s2 and computes the minimum jump cost between them. 
    It has been modified so that it can compute the MJC of time series that have arbitrarily spaced data points. 
    An early abandoning variable, dXYlimit, allows the user to specify a maximum dissimilarity that will cancel the
    computation.
    
    The time series are specified as follows:
    - s1 and s2 may be of different length. 
    - s1 and s2 may or may not have time information.
    - If one of the time series has time information, the other must also have it.
    
    A time series with no time information is just an array of values. The first element of the array corresponds to
    the earliest point in the time series. Example: s1 = [d_0, d_1, d_2, ...], where d_i is the ith value of the time
    series.
    A time series with time information must be a 2D array of shape (2, n). The data at index 0 are time
    data, and the data at index 1 is amplitude data.
    Example: s1 = [[t_0, t_1, t_2, ...], [d_0, d_1, d_2, ...]], where d_i is the ith value of the time series, and t_i
    is the time of the ith measurement. The time values may be integers or floats, and need not begin at 0.
    
    To visualize the algorithm, you may pass the variable showPlot=True. This will generate a plot with the two time
    series, and arrows signifying the jumps that the algorithm made when calculating the Minimum Jump Cost.
    
    ----EXECUTION SPEED----
    The time series are cast to numpy arrays. The checking and casting lowers execution speed. Therefore, an option to
    disable this checking and casting has been implemented. If you are absolutely sure that the time series s1 and s2
    are numpy.ndarray's of the format ([time data],[amplitude data]), you may pass the variable override_checks=True.
    
    As part of the calculation of the MJC, the algorithm calculates the standard deviations of the amplitude data, and
    the average length of time between each data point of s1 and s2. This lowers execution speed, but is required.
    However, if you know the standard deviations and/or the average time difference between data points of either
    (or both) s1 and s2 a priori, you may pass these as variables. They are named std_s1 and std_s2 and tavg_s1 and
    tavg_s2. None, any, or all of these may be passed.
    
    
    Parameters
    ----------
    s1              : Time series 1. 
    s2              : Time series 2. 
    dxy_limit       : Optional early abandoning variable. If the dissimilarity goes above this limit the computation is
        cancelled.
    beta            : Optional time jump cost. Defaults to 1. If 0, there is no cost associated with jumping forward.
    show_plot        : Optional. Defaults to False. If True, displays a plot that visualize the algorithms jump path at
        the end of the computation.
    std_s1          : Optional. Standard deviation of time series s1 amplitude data. See the section EXECUTION SPEED
        above for more information.
    std_s2          : Optional. Standard deviation of time series s2 amplitude data. See the section EXECUTION SPEED
        above for more information.
    tavg_s1         : Optional. Average value of time difference between consecutive data points of time series 1. See
        the section EXECUTION SPEED above for more information.
    tavg_s2         : Optional. Average value of time difference between consecutive data points of time series 2. See
        the section EXECUTION SPEED above for more information.
    return_args      : Optional. Causes the function to return the values for std_s1, std_s2, tavg_s1, tavg_s2, s1, and
        s2.
    override_checks  : Optional. Override checking if the supplied time series conform to the required format. See the
        section EXECUTION SPEED above for more information.
    
    Returns
    -------
    dXY         :   Cumulative dissimilarity measure.
    cancelled   :   Boolean. If True, the computation was cancelled as dXY reached dxy_limit.
    std_s1      :   Only returned if return_args=True. Value of std_s1 used in the computation.
    std_s2      :   Only returned if return_args=True. Value of std_s2 used in the computation.
    tavg_s1     :   Only returned if return_args=True. Value of tavg_s1 used in the computation.
    tavg_s2     :   Only returned if return_args=True. Value of tavg_s2 used in the computation.
    s1          :   Only returned if return_args=True. Value of s1 used in the computation.
    s2          :   Only returned if return_args=True. Value of s2 used in the computation.
    """
    dxy_limit = dxy_limit * beta
    # Check if the timeseries conform to the required format for the algorithm to work.
    # We allow the user to bypass checks to increase execution speed.
    if not override_checks:
        # Make sure arrays are numpy arrays. Cast to np.array if they are not.
        if not isinstance(s1, np.ndarray):
            s1 = np.array(s1)
        if not isinstance(s2, np.ndarray):
            s2 = np.array(s2)
        assert s1.ndim in [1, 2], "Series s1 must be either 1D or 2D."
        assert s2.ndim in [1, 2], "Series s2 must be either 1D or 2D."
        if s1.ndim != s2.ndim:
            raise ValueError(f"Both series s1 and s2 must have the same number of dimensions. "
                             f"s1 is {s1.ndim}D, s2 is {s2.ndim}D.")

        # Assert that data is numeric
        assert np.issubdtype(s1.dtype, np.number), f"Series s1 must be numeric, not {s1.dtype=}."
        assert np.issubdtype(s2.dtype, np.number), f"Series s2 must be numeric, not {s2.dtype=}."

        # Generate dummy time info so that the algorithm can work.
        if s1.ndim != 2:
            s1 = np.array([np.arange(s1.shape[0]), s1])
        if s2.ndim != 2:
            s2 = np.array([np.arange(s2.shape[0]), s2])


    s1shape = s1.shape
    s2shape = s2.shape

    # Plot the two time series, if show_plot is true.
    if show_plot:
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes()
        plot(s1[0], s1[1], 'bo', s1[0], s1[1], 'b')
        plot(s2[0], s2[1], 'ro', s2[0], s2[1], 'r')

    # Compute the standard deviations of s1 and s2 and the average delay between data points in s1 and s2 if they are
    # not provided.
    if std_s1 is None:
        std_s1 = stdev(s1[1])
    if std_s2 is None:
        std_s1 = stdev(s2[1])
    if tavg_s1 is None:
        tavg_s1 = np.average(np.ediff1d(s1[0]))
    if tavg_s2 is None:
        tavg_s2 = np.average(np.ediff1d(s2[0]))

    # We will only compute the MJC for the parts of s1 and s2 that overlap.
    # Get start index of the dataset that starts the earliest.
    s1start = s1[0, 0]
    s2start = s2[0, 0]
    t1 = np.searchsorted(s1[0], s2start, side="left") if s1start < s2start else 0
    t2 = np.searchsorted(s2[0], s1start, side="left") if s1start > s2start else 0

    # Get end index of the dataset that ends the latest.
    s1end = s1[0, -1]
    s2end = s2[0, -1]
    s1Length = np.searchsorted(s1[0], s2end, side="right") + 1 if s1end > s2end else s1shape[1]
    s2Length = np.searchsorted(s2[0], s1end, side="right") + 1 if s1end < s2end else s2shape[1]

    assert s2[0, s2Length - 1] - s1[0, t1] > 0 and s1[0, s1Length - 1] - s2[0, t2] > 0, 'Time series not overlapping!'
    # Compute time jump cost constants
    phi1 = beta * 4 * std_s1 / (s1[0, s1Length - 1] - s1[0, t1])
    phi2 = beta * 4 * std_s1 / (s2[0, s2Length - 1] - s2[0, t2])

    # Initiate the cumulative dissimilarity measure dXY.
    dXY = 0

    # Begin computation of the cumulative dissimilarity measure.
    while t1 < s1Length and t2 < s2Length:
        c, t1, t2 = cmin(s1, t1, s2, t2, s2Length, phi2, tavg_s2, beta)
        if show_plot:
            ax.arrow(s1[0, t1 - 1], s1[1, t1 - 1], s2[0, t2 - 1] - s1[0, t1 - 1], s2[1, t2 - 1] - s1[1, t1 - 1],
                     width=0.005, head_width=0.1)
        dXY += c
        if dXY >= dxy_limit:
            return dXY, True
        if t1 >= s1Length or t2 >= s2Length:
            break
        c, t2, t1 = cmin(s2, t2, s1, t1, s1Length, phi1, tavg_s1, beta)
        if show_plot:
            ax.arrow(s2[0, t2 - 1], s2[1, t2 - 1], s1[0, t1 - 1] - s2[0, t2 - 1], s1[1, t1 - 1] - s2[1, t2 - 1],
                     width=0.005, head_width=0.1)
        dXY += c
        if dXY >= dxy_limit:
            if return_args:
                return dXY, True, std_s1, std_s2, tavg_s1, tavg_s2, s1, s2
            else:
                return dXY, True

    if show_plot:
        plt.show()

    if return_args:
        return dXY, False, std_s1, std_s2, tavg_s1, tavg_s2, s1, s2
    else:
        return dXY, False


def cmin(s1, t1, s2, t2, N, phi, tavg, beta):
    cmin = inf
    d = 0
    dmin = 0
    s2t = s2[0, max(t2 - 1, 0)]  # Start time of series 2
    fjump = s2[0, t2] - s2t - tavg  # First forced jump length.
    while t2 + d < N:
        if d == 0 and fjump > 0:
            c = pow((phi * (max(s2[0, t2] - s2t - tavg, 0)) / beta), 2)  # The beta of forced jumps are set to 1.
        else:
            # We have replaced d with the time difference between the current point and the previous point
            # that was jumped to, and we also subtract the average jump length to make an "average" first jump costless.
            c = pow((phi * (max(s2[0, t2] - s2t - tavg, 0))), 2)
        if c >= cmin:
            if t2 + d > t1:
                break
        else:
            c += pow((s1[1, t1] - s2[1, t2 + d]), 2)
            if c < cmin:
                cmin = c
                dmin = d
        d += 1
    t1 += 1
    t2 += dmin + 1
    return cmin, t1, t2
