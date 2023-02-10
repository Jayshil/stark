"""
Created on Fri Aug  9 22:33:50 2019

@author: Alexis Brandeker (Initial Version)
@author: Jayshil A. Patel (Modified to use with data cube)
"""
import numpy as np
from multiprocessing import Pool
from scipy.interpolate import LSQUnivariateSpline, LSQBivariateSpline
import warnings
import time

def aperture_extract(frame, variance, ord_pos, ap_rad, uniform = False):
    """ Simple aperture extraction of the spectrum

    Given the 2D `frame` of the data, and its `variance`, this function
    extracts spectrum by simply adding up values of pixels along slit.

    Parameters
    ----------
    frame : ndarray
        2D data frame from which the spectrum is to be extracted
    variance : ndarray
        The noise image of the same format as frame, specifying
        the variance of each pixel.
    ord_pos : ndarray
        Array defining position of the trace
    ap_rad : float
        Radius of the aperture around the order position
    uniform : bool, optional
        Boolean on whether the slit is uniformally lit or not.
        If not then it will simply sum up counts in the aperture
        else average the counts and multiply by slit-length.
        Default is False.

    Returns
    -------
    spec : ndarray
        Array containing extracted spectrum for each order and each column.
    var : ndarray
        Variance array for `spec`, the same shape as `spec`.
    """
    nslitpix = ap_rad*2
    ncols = frame.shape[1]
    spec = np.zeros(ncols)
    var = np.zeros(ncols)

    for col in range(ncols):
        if ord_pos[col] < 0 or ord_pos[col] >= frame.shape[0]:
            continue
        i0 = int(round(ord_pos[col] - ap_rad))
        i1 = int(round(ord_pos[col] + ap_rad))

        if i0 < 0:
            i0 = 0
        if i1 >= frame.shape[0]:
            i1 = frame.shape[0] - 1
        if uniform:
            spec[col] = np.mean(frame[i0:i1,col])*nslitpix
            var[col] = np.mean(variance[i0:i1,col])*nslitpix
        else:
            spec[col] = np.sum(frame[i0:i1,col])
            var[col] = np.sum(variance[i0:i1,col])
    return spec, var