from math import pow, inf

def MJC(X, Y):
    """
    Minimum Jump Cost (MJC) dissimiliarity algorithm.
    This algorithm implements the MJC algorithm by Joan Serra and Josep Lluis Arcos (2012).
    
    Parameters
    ----------
    X = Time series 1
    Y = Time series 2
    
    Returns
    -------
    dXY :   Cumulative dissimilarity measure
    """
    
    tx = 0
    ty = 0
    dXY = 0
    
    xLength = len(X)
    yLength = len(Y)
    
    while tx<xLength and ty<yLength:
        c, tx, ty = cmin(X, tx, Y, ty, yLength)
        dXY += c
        if tx>=xLength or ty>=yLength:
            break
        c, tx, ty = cmin(Y, ty, X, tx, xLength)
        dXY += c
    return dXY
    
    
    
def cmin(X,tx,Y,ty, N):
    phi = 1
    cmin = inf
    d = 0
    dmin = 0
    while ty + d < N:
        c = pow((phi*d),2)
        if c >= cmin:
            if ty + d > tx:
                break
        else:
            c += pow((X[tx]-Y[ty]+d), 2)
            if c < cmin:
                cmin = c
                dmin = d
        d += 1
    tx += 1
    ty += dmin + 1
    return cmin, tx, ty