"""
Statistics Module

Various bivariate statistics functions typically used as Goodness Of Fit measures for
hydrologic models.

Functions modules contained here are designed to be called on pairs of DataFrames,
(observed and predicted) with equivalent columns.

All functions should work in various scenarios, such as:

* Dataframes with columns for different constituents,
* Dataframes with columns for different sites
* Daily, monthly, annual time steps
* Other timeperiods, such data aggregated to water year

Broadly speaking, these statistics will work with data that
"""

def intersect(obs,pred):
    """
    Return the input pair of dataframes (obs,pred) with a common index made up of the intersection of
    the input indexes
    """
    if hasattr(obs,'intersect'):
        return obs.intersect(pred)
    idx = obs.index.intersection(pred.index)
    return obs.ix[idx],pred.ix[idx]

def nse(obs,pred):
    """
    Nash-Sutcliffe Efficiency
    """
    obs,pred = intersect(obs,pred)
    pred = pred.ix[obs.index] # Filter values not present in
    numerator = ((obs-pred)**2).sum()
    denominator = ((obs-obs.mean())**2).sum()
    return 1 - numerator/denominator
nse.perfect=1
nse.maximise=True

def PBIAS(obs,pred):
    obs,pred = intersect(obs,pred)
    top = (obs-pred).sum()
    bottom = obs.sum()
    return (top/bottom)*100
    
def rsr (obs,pred):
    obs,pred = intersect(obs,pred)
    rmse = (((obs-pred)**2).sum())**(1/2)
    stdev_obs = (((obs-obs.mean())**2).sum())**(1/2)
    return rmse/stdev_obs
# RSR could alternatively be calculated as: RSR = (1-nse)**(1/2)
 
def ppmc (obs,pred):
    """
    Pearson Product Moment Correlation (PPMC)
    """
    obs,pred = intersect(obs,pred)
    top = ((obs-obs.mean())*(pred-pred.mean())).sum()
    bottom = (((obs-obs.mean())**2).sum())**(1/2)*(((pred-pred.mean())**2).sum())**(1/2)
    return top/bottom
    
def ioad (obs,pred):
    """
    ioad - Index of Agreement
    """
    obs,pred = intersect(obs,pred)
    top = ((obs-pred)**2).sum()
    bottom = (((pred-obs.mean()).abs()+(obs+obs.mean()).abs())**2).sum()
    return 1 - top/bottom
    
def rae (obs,pred):
    """
    Relative Absolute Error (RAE)
    """
    obs,pred = intersect(obs,pred)
    top = ((obs-pred).abs()).sum()
    bottom = ((obs-obs.mean()).abs()).sum()
    return top/bottom
  