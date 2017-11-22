from math import pow, inf
from statistics import stdev
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, grid, show
import numpy as np

def MJC(s1, s2, dXYlimit=inf, beta=1, overrideChecks=False, showPlot=False, stds1=None, stds2=None):
    """
    Minimum Jump Cost (MJC) dissimiliarity algorithm.
    This algorithm implements the MJC algorithm by Joan Serra and Josep Lluis Arcos (2012). This algorithm was shown to outperform the Dynamic Time Warp (DTW) dissimilarity algorithm on several datasets.
    
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
    
    ----EXECUTION SPEED----
    The time series are cast to numpy arrays. The checking and casting lowers execution speed. Therefore, an option to disable this 
    checking and casting has been implemented. If you are absolutely sure that the time series s1 and s2 are numpy.ndarray's of the 
    format ([time data],[amplitude data]), you may pass the variable overrideChecks=True.
    
    As part of the calculation of the MJC, the algorithm calculates the standard deviations of s1 and s2. This lowers execution speed,
    but is required. However, if you know the standard deviations of either (or both) s1 and s2 a priori, you may pass these as variables.
    They are named stds1 and stds2.
    
    
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
    """
    
    #Check if the timeseries conform to the required format for the algorithm to work.
    #We allow the user to bypass checks to speed up execution speed.
    if not overrideChecks:
        #Make sure arrays are numpy arrays. Cast to np.array if they are not.
        if not isinstance(s1, np.ndarray):
            s1 = np.array(s1)
        if not isinstance(s2, np.ndarray):
            s2 = np.array(s2)   
        s1shape = s1.shape
        s2shape = s2.shape
        #Check whether time series s1 and s2 have time information.
        #Also verify their shapes and formats are correct
        hasT = False
        if len(s1shape) != 1 or len(s2shape) != 1:
            if len(s1shape) != 2 or len(s2shape) != 2:
                raise RuntimeError("The time series s1 and s2 must both be a list of two lists. The first being the time information of the series, and the second its values.")
            #We now know they have the right dimensions.
            hasT = True
        else:
            #We generate dummy time info so that the algorithm can work.
            s1 = np.array([np.arange(s1shape[0]),s1])
            s2 = np.array([np.arange(s2shape[0]),s2])
            s1shape = s1.shape
            s2shape = s2.shape        
    else:
        s1shape = s1.shape
        s2shape = s2.shape
        
    #Plot the two time series, if showPlot is true.
    if showPlot:
        fig=plt.figure(figsize=(13,7))
        ax = plt.axes()
        plot(s1[0], s1[1], 'bo', s1[0], s1[1], 'b')
        plot(s2[0], s2[1], 'ro', s2[0], s2[1], 'r')
            
    #We allow for the user to precompute the standard deviations of s1 and s2.
    if stds1==None:
        stds1 = stdev(s1[1])
    if stds2==None:
        stds2 = stdev(s2[1])
    #Compute time jump cost constants
    phi1 = beta*4*stds1/(s1[0,-1]-s1[0,0])
    phi2 = beta*4*stds2/(s2[0,-1]-s2[0,0])
    #Get timeseries lengths.
    s1Length = s1shape[1]
    s2Length = s2shape[1]
    #Initiate step counters and the cumulative dissimilarity measure dXY.
    t1 = 0
    t2 = 0
    dXY = 0
    
    #Begin computation of the cumulative dissimilarity measure.
    while t1<s1Length and t2<s2Length:
        c, t1, t2 = cmin(s1, t1, s2, t2, s2Length, phi2)
        if showPlot:
            ax.arrow(s1[0,t1-1], s1[1,t1-1], s2[0,t2-1]-s1[0,t1-1], s2[1,t2-1]-s1[1,t1-1], width=0.005, head_width=0.1)
        dXY += c
        if dXY>=dXYlimit:
            return dXY, True
        if t1>=s1Length or t2>=s2Length:
            break
        c, t2, t1 = cmin(s2, t2, s1, t1, s1Length, phi1)
        if showPlot:
            ax.arrow(s2[0,t2-1], s2[1,t2-1], s1[0,t1-1]-s2[0,t2-1], s1[1,t1-1]-s2[1,t2-1], width=0.005, head_width=0.1)
        dXY += c
        if dXY>=dXYlimit:
            return dXY, True
            
    if showPlot:
        plt.show()
        
    return dXY, False
    
    
    
def cmin(s1,t1,s2,t2,N,phi):
    cmin = inf
    d = 0
    dmin = 0
    s2t = s2[0, t2] #Start time of series 2
    while t2 + d < N:
        c = pow((phi*(s2[0, t2]-s2t)),2) #We have replaced d with the time difference between the first and current series2 datapoint.
        if c >= cmin:
            if t2 + d > t1:
                break
        else:
            c += pow((s1[1, t1]-s2[1, t2+d]), 2)
            if c < cmin:
                cmin = c
                dmin = d
        d += 1
    t1 += 1
    t2 += dmin + 1
    return cmin, t1, t2