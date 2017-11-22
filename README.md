# Minimum Jump Cost algorithm

This python code implements the Minimum Jump Cost (MJC) algorithm dissimilarity algorithm devised by Joan Serra and Josep Lluis Arcos in 2012. MJC was shown to be computationally fast, and sometimes even more accurate than Dynamic Time Warp (DTW). You can read the paper here: 
https://www.iiia.csic.es/sites/default/files/4584.pdf

## How to install

Open a command window or bash shell. Run `pip install mjc`. 

## How to use
Example: 
```
from mjc import MJC

series_1 = [1,3,7,2]
series_2 = [3,1,2.2,0.1]

dXY, abandoned = MJC(series_1, series_2)

print("The dissimilarity of series 1 and series 2 is {}".format(dXY))
```
There are several options for increasing the execution speed of this algorithm. They are detailed in the next section.

### More information (from the function docs)
This function takes two time series s1 and s2 and computes the minimum jump cost between them. 
It has been modified so that it can compute the MJC of time series that have arbitrarily spaced data points. 
An early abandoning variable, dXYlimit, allows the user to specify a maximum dissimilarity that will cancel the computation.

The time series are specified as follows:
- s1 and s2 may be of different length. 
- s1 and s2 may or may not have time information.
- If one of the time series has time information, the other must also have it.
    
A time series with no time information is just an array of values. The first element of the array corresponds to the earliest point in the time series. Example: s1 = [d_0, d_1, d_2, ...], where d_i is the ith value of the time series.
A tme series with time information must be an array of two arrays. The first holds the time information of each point (i.e. the x-axis values), and the other array holds the amplitude data (i.e. the y-axis values). Example: s1 = [[t_0, t_1, t_2, ...], [d_0, d_1, d_2, ...]], where d_i is the ith value of the time series, and t_i is the time of the ith measurement.
The time values may be integers or floats, and need not begin at 0.

To visualize the algorithm, you may pass the variable showPlot=True. This will generate a plot with the two time series, and arrows signifying the jumps that the algorithm made when calculating the Minimum Jump Cost.    

#### EXECUTION SPEED
The time series are cast to numpy arrays. The checking and casting lowers execution speed. Therefore, an option to disable this 
checking and casting has been implemented. If you are absolutely sure that the time series s1 and s2 are numpy.ndarray's of the 
format ([time data],[amplitude data]), you may pass the variable overrideChecks=True.

As part of the calculation of the MJC, the algorithm calculates the standard deviations of s1 and s2. This lowers execution speed,
but is required. However, if you know the standard deviations of either (or both) s1 and s2 a priori, you may pass these as variables.
They are named stds1 and stds2.

```
Parameters
----------
s1              : Time series 1. 
s2              : Time series 2. 
dXYlimit        : Optional early abondoning variable. If the dissimilarity goes above this limit the 
computation is cancelled.
beta            : Optional time jump cost. Defaults to 1. If 0, there is no cost associated with jumping forward.
overrideChecks  : Optional. Override checking if the supplied time series conform to the required format. See the section EXECUTION SPEED above for more information.
showPlot        : Optional. Defaults to False. If True, displays a plot that visualize the algorithms jump path at the end of the computation.
stds1           : Optional. Standard deviation of time series s1 amplitude data. See the section EXECUTION SPEED above for more information.
stds2           : Optional. Standard deviation of time series s2 amplitude data. See the section EXECUTION SPEED above for more information.
    
Returns
-------
dXY         :   Cumulative dissimilarity measure.
cancelled   :   Boolean. If True, the computation was cancelled as dXY reached dXYlimit.
```
