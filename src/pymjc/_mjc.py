from statistics import stdev
import matplotlib.pyplot as plt
import numpy as np


def dmjc(s1, s2, dxy_limit=np.inf, beta=1., show_plot=False, std_s1=None, std_s2=None, tavg_s1=None,
         tavg_s2=None, override_checks=False):
    """
    This is the symmetrized version of the Minimum Jump Cost dissimilarity measure. Depending on whether we start at s1
    or s2 we will obtain different values. This computes both and returns the lowest value.

    See pymjc() for definition of variables and return values. """
    dxy_a, abandoned_a = mjc(s1, s2, dxy_limit, beta, show_plot, std_s1, std_s2, tavg_s1, tavg_s2, return_args=True,
                             override_checks=override_checks)
    dxy_b, abandoned_b = mjc(s2, s1, dxy_limit, beta, show_plot, std_s2, std_s1, tavg_s2, tavg_s1,
                             override_checks=override_checks)
    return min(dxy_a, dxy_b), abandoned_a and abandoned_b


def mjc(s1, s2, dxy_limit=np.inf, beta=1., show_plot=False, std_s1=None, std_s2=None, tavg_s1=None, tavg_s2=None,
        return_args=False, override_checks=False):
    """
    Minimum Jump Cost (MJC) dissimiliarity measure.
    This algorithm implements the MJC algorithm devised by Joan Serra and Josep Lluis Arcos (2012). This algorithm was
    shown to outperform the Dynamic Time Warp (DTW) dissimilarity algorithm on several datasets.

    pymjc() takes two time series s1 and s2 and computes the minimum jump cost between them.
    It has been modified so that it can compute the MJC of time series that have arbitrarily spaced data points,
    different sampling rates, and non-overlapping regions.

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

    ---- PERFORMANCE ----
    The time series are cast to numpy arrays. The checking and casting lowers execution speed. Therefore, an option to
    disable this checking and casting has been implemented. If you are certain that the time series s1 and s2
    are numpy.ndarray's of the format [[time data],[amplitude data]], you may pass the variable override_checks=True.

    The algorithm locates the overlapping region between the two timeseries. This step is skipped if the first and last
    timestamps are equal between the two timeseries. If your data has no time data, it is skipped if there is the same
    number of samples in each timeseries.

    As part of the calculation of the MJC, the algorithm calculates the standard deviations of the amplitude data, and
    the average sampling period of s1 and s2. This lowers execution speed, but is required.
    However, if you know the standard deviations and/or the average time difference between data points of either
    (or both) s1 and s2 a priori, you may pass these as variables. They are named std_s1 and std_s2 and tavg_s1 and
    tavg_s2. Any number of these may be passed. The ones which are not passed will be calculated.


    Parameters
    ----------
    s1              : numpy ndarray. Time series 1.
    s2              : numpy ndarray. Time series 2.
    dxy_limit       : Optional float. Early abandoning variable. If the dissimilarity measure exceeds this limit the
        computation is cancelled. Default infinity.
    beta            : Optional float. Time jump cost. If 0, there is no cost associated with jumping forward. Default 1.
    show_plot       : Optional bool. If True, displays a plot that visualize the algorithms jump path. Default False.
    std_s1          : Optional float. Standard deviation of time series s1. See the section PERFORMANCE above for more
        information.
    std_s2          : Optional float. Standard deviation of time series s2. See the section PERFORMANCE above for more
        information.
    tavg_s1         : Optional float. Average sampling period of time series 1. See the section PERFORMANCE above for
        more information. If your data does not have time information, this value can be set to 1.
    tavg_s2         : Optional float. Average sampling period of time series 2. See the section PERFORMANCE above for
        more information. If your data does not have time information, this value can be set to 1.
    return_args      : Optional bool. If True, returns the values for std_s1, std_s2, tavg_s1, tavg_s2, s1, and s2.
    override_checks  : Optional bool. Override checking and casting if the supplied time series are known to conform to
        the required format. See the section PERFORMANCE above for more information.

    Returns
    -------
    d_xy         :   Cumulative dissimilarity measure.
    cancelled   :   Boolean. If True, the computation was cancelled as d_xy reached dxy_limit.
    std_s1      :   Only returned if return_args=True. Value of std_s1 used in the computation.
    std_s2      :   Only returned if return_args=True. Value of std_s2 used in the computation.
    tavg_s1     :   Only returned if return_args=True. Value of tavg_s1 used in the computation.
    tavg_s2     :   Only returned if return_args=True. Value of tavg_s2 used in the computation.
    s1          :   Only returned if return_args=True. Value of s1 used in the computation.
    s2          :   Only returned if return_args=True. Value of s2 used in the computation.
    """
    assert beta >= 0, "'beta' must be greater or equal to 0."
    assert dxy_limit >= 0, "'dxy_limit' must be greater or equal to 0."
    s1, s2 = check_input(s1, s2, override_checks)

    # Get views of the overlapping region and their lengths
    s1_ovrlap, s2_ovrlap = get_overlapping_region(s1, s2)

    # Compute the standard deviations of s1 and s2 and the average time between data points in s1 and s2 if they are
    # not provided. In the original paper it is unclear how the standard deviation is defined. I am assuming it is a
    # mean of the standard deviation of both timeseries.
    std_s1, std_s2, tavg_s1, tavg_s2 = analyse_timeseries(std_s1, std_s2, tavg_s1, tavg_s2, s1_ovrlap, s2_ovrlap)
    std_mean = (std_s1 + std_s2) / 2

    # Compute the jump cost d_xy
    d_xy, jumps = jump_cost(s1_ovrlap, s2_ovrlap, dxy_limit, beta, std_mean, tavg_s1, tavg_s2)
    limit_reached = d_xy >= dxy_limit

    # Plot the two time series and the jumps between them, if show_plot is true.
    if show_plot:
        plot(s1_ovrlap, s2_ovrlap, jumps, d_xy)

    if return_args:
        return d_xy, limit_reached, std_s1, std_s2, tavg_s1, tavg_s2, s1, s2
    else:
        return d_xy, limit_reached


def analyse_timeseries(std_s1, std_s2, tavg_s1, tavg_s2, s1, s2):
    if std_s1 is None:
        std_s1 = stdev(s1[1])
    else:
        assert std_s1 > 0, "'std_s1' must be greater than 0."
    if std_s2 is None:
        std_s2 = stdev(s2[1])
    else:
        assert std_s2 > 0, "'std_s2' must be greater than 0."
    if tavg_s1 is None:
        tavg_s1 = np.average(np.ediff1d(s1[0]))
    else:
        assert tavg_s1 > 0, "'tavg_s1' must be greater than 0."
    if tavg_s2 is None:
        tavg_s2 = np.average(np.ediff1d(s2[0]))
    else:
        assert tavg_s2 > 0, "'tavg_s2' must be greater than 0."

    return std_s1, std_s2, tavg_s1, tavg_s2


def get_overlapping_region(s1, s2):
    """
    Finds the region of s1 and s2 which are overlapping in time, and returns a view of this region for s1 and s2.

    Parameters
    ----------
    s1  : np.ndarray. Data series 1.
    s2  : np.ndarray. Data series 2.

    Returns
    -------
    s1_overlapping  : A view of the input array s1 which overlaps with s2.
    s2_overlapping  : A view of the input array s2 which overlaps with s1.
    """
    # Fast path in case there is full overlap
    if s1[0, 0] == s2[0, 0] and s1[0, -1] == s2[0, -1]:
        return s1, s2

    s1start = s1[0, 0]
    s2start = s2[0, 0]
    s1_start_idx = np.searchsorted(s1[0], s2start, side="left") if s1start < s2start else 0
    s2_start_idx = np.searchsorted(s2[0], s1start, side="left") if s1start > s2start else 0

    s1end = s1[0, -1]
    s2end = s2[0, -1]
    s1_end_idx = np.searchsorted(s1[0], s2end, side="right") if s1end > s2end else s1.shape[1]
    s2_end_idx = np.searchsorted(s2[0], s1end, side="right") if s1end < s2end else s2.shape[1]

    assert s2[0, s2_end_idx - 1] - s1[0, s1_start_idx] > 0 and s1[0, s1_end_idx - 1] - s2[0, s2_start_idx] > 0, \
        'The time series s1 and s2 have no overlapping regions.'
    return s1[:, s1_start_idx:s1_end_idx], s2[:, s2_start_idx:s2_end_idx]


def check_input(s1, s2, override_checks):
    """
    Performs checks on input data to verify that they conform to the required format for the MJC algorithm.

    Parameters
    ----------
    s1              : List or np.ndarray. Data series 1.
    s2              : List or np.ndarray. Data series 1.
    override_checks : Bool. Whether to perform checks or not.

    Returns
    -------
    s1  : np.ndarray. Checked and optionally cast to np.ndarray of originally a List.
    s2  : np.ndarray. Checked and optionally cast to np.ndarray of originally a List.
    """
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
        assert np.issubdtype(s1.dtype, np.number), f"Series s1 must be numeric, not {s1.dtype}."
        assert np.issubdtype(s2.dtype, np.number), f"Series s2 must be numeric, not {s2.dtype}."

        # Generate dummy time info so that the algorithm can work.
        if s1.ndim != 2:
            s1 = np.array([np.arange(s1.shape[0]), s1])
        if s2.ndim != 2:
            s2 = np.array([np.arange(s2.shape[0]), s2])
    return s1, s2


def jump_cost(x, y, dxy_limit, beta, std, tavg_x, tavg_y):
    """
    Calculates the minimum jump cost d_xy between timeseries x and y.
    Parameters
    ----------
    x           : np.ndarray of shape (2,n). Row 0 is time, row 1 is data.
    y           : np.ndarray of shape (2,m). Row 0 is time, row 1 is data.
    dxy_limit   : Early abandoning variable. Computation is cancelled if the jump cost exceeds this value.
    beta        : Time jump cost.
    std         : Timeseries standard deviation, used in calcilation of the time advancement cost phi
    tavg_x      : Average sample period of timeseries x.
    tavg_y      : Average sample period of timeseries y.

    Returns
    -------
    d_xy    : The minimum jump cost d_xy
    jumps   : np.ndarray of all the jumps in the format [idx_x, idx_y] for even jumps, and [idx_y, idx_x] for odd jumps.
    """
    x_len = x.shape[1]
    y_len = y.shape[1]

    # Compute time advancement cost phi.
    # Due to the potential different sampling frequency of s1 and s2, we must compute two phis.
    phi1 = beta * 4 * std / x_len
    phi2 = beta * 4 * std / y_len

    # Initiate the cumulative dissimilarity measure d_xy.
    d_xy = 0

    # Begin computation of the cumulative dissimilarity measure.
    jumps = np.empty(shape=(min(x_len, y_len), 2), dtype=int)
    dxy_limit = dxy_limit * beta
    idx_x = idx_y = 0
    n = 0
    while idx_x < x_len and idx_y < y_len:
        c, idx_x, idx_y, _idx_x, _idx_y = cmin(x, idx_x, y, idx_y, y_len, phi2, tavg_x, tavg_y)
        d_xy += c
        jumps[n] = [_idx_x, _idx_y]
        n += 1

        # Break out of loop if end of datasets have been reached or the d_xy limit has been crossed.
        if idx_x >= x_len or idx_y >= y_len or d_xy >= dxy_limit:
            break

        c, idx_y, idx_x, _idx_y, _idx_x = cmin(y, idx_y, x, idx_x, x_len, phi1, tavg_y, tavg_x)
        d_xy += c
        jumps[n] = [_idx_y, _idx_x]
        n += 1

        # Break out of loop if the d_xy limit has been crossed
        if d_xy >= dxy_limit:
            break

    jumps = jumps[:n]
    return d_xy, jumps


def cmin(x, idx_x, y, idx_y, n, phi, t_avg_x, t_avg_y):
    c_min = np.inf
    d = 0
    dmin = 0
    time_y = y[0, idx_y]  # Start time of y
    time_x = x[0, idx_x]  # Start time of x
    while idx_y + d < n:
        # We have replaced d in the original paper pseudocode with the normalized time difference dt
        current_time_y = y[0, idx_y + d]
        dt = (current_time_y - time_y) / t_avg_y
        c = pow(phi * dt, 2)

        if c >= c_min:
            if current_time_y > time_x:
                break
        else:
            c += pow((x[1, idx_x] - y[1, idx_y + d]), 2)
            if c < c_min:
                c_min = c
                dmin = d
        d += 1
    _idx_x = idx_x
    _idx_y = idx_y
    idx_x += max(1, int(t_avg_y / t_avg_x))
    idx_y += dmin + max(1, int(t_avg_x / t_avg_y))
    return c_min, idx_x, idx_y, _idx_x, _idx_y


def plot(s1, s2, jumps, d_xy):
    plt.figure(figsize=(13, 7))
    ax = plt.axes()
    plt.plot(s1[0], s1[1], 'bo', s1[0], s1[1], 'b')
    plt.plot(s2[0], s2[1], 'ro', s2[0], s2[1], 'r')
    arrow_scale = max(s1[0, -1] - s1[0, 0], s2[0, -1] - s2[0, 0])
    plt.title(f"Dissimilarity measure dXY (total jump cost): {d_xy:.3f}")
    for n, jump in enumerate(jumps):
        p0 = s1[:, jump[0]] if n % 2 == 0 else s2[:, jump[0]]
        p1 = s2[:, jump[1]] if n % 2 == 0 else s1[:, jump[1]]
        ax.arrow(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1], width=0.002 * arrow_scale)
    plt.show()
