![LICENCE](https://img.shields.io/github/license/nup002/mjc)
[![Flake8](https://github.com/nup002/mjc/actions/workflows/flake8.yml/badge.svg)](https://github.com/nup002/mjc/actions/workflows/flake8.yml)
[![PyTest](https://github.com/nup002/mjc/actions/workflows/PyTest.yml/badge.svg)](https://github.com/nup002/mjc/actions/workflows/PyTest.yml)
# Minimum Jump Cost dissimilarity measure in Python

This package implements the Minimum Jump Cost (MJC) dissimilarity measure devised by Joan Serra and Josep Lluis Arcos in 2012. MJC was shown to be computationally fast, and sometimes even more accurate than Dynamic Time Warp (DTW). You can read the paper here: 
https://www.iiia.csic.es/sites/default/files/4584.pdf.

This package can compute the MJC for timeseries with different sampling rates, arbitrarily spaced data points, and 
non-overlapping regions.

## How to install
The 'mjc' package on PyPi is non-functional. Instead, download this repository directly. I am working on publishing it to PyPi at this moment, and it should be ready within a week (before 10 December 2022).


## How to use
Example: 
```
from mjc import mjc
import numpy as np

series_1 = np.array([1,2,3,4])
series_2 = np.array([2,3,4,5])

d_xy, abandoned = mjc(series_1, series_2, show_plot=True)

print(f"The MJC dissimilarity of series 1 and series 2 is {d_xy}")
```
There are some options for reducing the computational load of this algorithm. They are detailed in the next section.

## More detailed information
The time series s1 and s2 are specified as follows:
- They may be python Lists or numpy.ndarrays
- They may be of different length.
- They may or may not have time information.
- If one of the time series has time information, the other must also have it.
- Their datatype may be floats or integers.

A time series with no time information is just a list of values. The first element of the list corresponds to
the earliest point in the time series.<br>
Example: `s1 = [d₀, d₁, d₂, ...]`, where `dᵢ` is the i-th value of the time series.

A time series with time information must be a 2D array of shape (2, n). The data at index 0 are time
data, and the data at index 1 is amplitude data.<br>
Example: `s1 = [[t₀, t₁, t₂, ...], [d₀, d₁, d₂, ...]]`, where `tᵢ` is the time of the i-th measurement. The time 
values may be integers or floats, and need not begin at 0.

To visualize the algorithm, you may pass the variable `show_plot=True`. This will generate a plot with the two time
series, and arrows signifying the jumps that the algorithm made when calculating the Minimum Jump Cost.

To stop the algorithm early, pass a value for `dxy_limit`. If the dissimilarity measure exceeds this value during 
computation, it is abandoned.


### Performance
The time series are cast to numpy arrays. The checking and casting lowers execution speed. Therefore, an option to
disable this checking and casting has been implemented. If you are certain that the time series s1 and s2
are numpy.ndarray's of the format `[[time data],[amplitude data]]`, you may pass the variable `override_checks=True`.

As part of the calculation of the MJC, the algorithm calculates the standard deviations of the amplitude data, and
the average sampling period of s1 and s2. This lowers execution speed, but is required.
However, if you know the standard deviations and/or the average time difference between data points of either
(or both) s1 and s2 a priori, you may pass these as variables. They are named std_s1 and std_s2 and tavg_s1 and
tavg_s2. Any number of these may be passed. The ones which are not passed will be calculated.

mjc() input parameters:
```
s1              : numpy ndarray | List. Time series 1.
s2              : numpy ndarray | List. Time series 2.
dxy_limit       : Optional float. Early abandoning variable.
beta            : Optional float. Time jump cost. 
show_plot       : Optional bool. If True, displays a plot that visualize the algorithms jump path. Default False.
std_s1          : Optional float. Standard deviation of time series s1.
std_s2          : Optional float. Standard deviation of time series s2.
tavg_s1         : Optional float. Average sampling period of time series 1.
tavg_s2         : Optional float. Average sampling period of time series 2. 
return_args     : Optional bool. If True, returns the values for std_s1, std_s2, tavg_s1, tavg_s2, s1, and s2.
override_checks : Optional bool. Override checking and casting
```
