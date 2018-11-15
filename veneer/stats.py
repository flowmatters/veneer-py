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

import numpy as np

def compNFDC(obs, pred):
	"""
	Composite of NSE and NSE of log FDC
	"""
	x = 0.5
	NSE = nse(obs, pred)
	LOGFDC = NSElogFDC(obs, pred)
	return x*NSE+(1-x)*LOGFDC
compNFDC.perfect=1
compNFDC.maximise=True

def sort(obs, pred):
	"""
	Return sorted obs and pred time series'
	"""
	obs = obs.sort_values(ascending=True)
	pred = pred.sort_values(ascending=True)
	return obs,pred

def log_10(obs, pred):
	"""
	Return log of obs and pred time series'
	"""
	obs = np.log10(obs+0.01)
	pred = np.log10(pred+0.01)
	return obs,pred

def NSElogFDC(obs,pred):
	"""
	Return the Nash-Sutcliffe Efficiency of flow duration of log data
	"""
	obs,pred = intersect(obs,pred)
	obs,pred = sort(obs,pred)
	obs,pred = log_10(obs,pred)
	numerator = ((obs.values-pred.values)**2).sum()
	denominator = ((obs.values-obs.mean())**2).sum()
	return 1 - numerator/denominator
NSElogFDC.perfect=1
NSElogFDC.maximise=True

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
  