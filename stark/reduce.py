"""
Created on Thu Feb 16 16:36:09 2023

@author: Jayshil A. Patel (Initial version)
"""
import numpy as np
from skimage import restoration
from scipy.optimize import minimize
from scipy.interpolate import LSQUnivariateSpline
from tqdm import tqdm
from scipy.signal import medfilt2d

def col_by_col_bkg_sub(frame, mask):
    """To perform column by column background subtraction

    Given a frame and a mask on source, this function perform background
    subtraction at column level. The subtracted background is median of
    unmasked data.

    The original version of this function was written for `jwebbinars`:
    https://github.com/spacetelescope/jwebbinar_prep/blob/main/tso_session/ExploringTSOProducts.ipynb

    Parameters
    ----------
    frame : ndarray
        2D data frame of the data with shape [nrows, ncols]
    mask : ndarray
        Mask array with the same size as `frame`. Only points with 1 will be
        considered to estimate background.
    
    Returns
    -------
    corrected_image : ndarray
        Background corrected image.
    """
    corrected_image = np.ones(frame.shape)

    for i in range(frame.shape[1]):
        idx = np.where(mask[:,i] == 1)
        corrected_image[:, i] = frame[:,i] - np.nanmedian(frame[idx,i])
    return corrected_image 

def row_by_row_bkg_sub(frame, mask):
    """To perform row by row background subtraction

    Given a frame and a mask on source, this function perform background
    subtraction at row level. The subtracted background is median of
    unmasked data.

    The original version of this function was written for `jwebbinars`:
    https://github.com/spacetelescope/jwebbinar_prep/blob/main/tso_session/ExploringTSOProducts.ipynb

    Parameters
    ----------
    frame : ndarray
        2D data frame of the data with shape [nrows, ncols]
    mask : ndarray
        Mask array with the same size as `frame`. Only points with 1 will be
        considered to estimate background.
    
    Returns
    -------
    corrected_image : ndarray
        Background corrected image.
    """
    corrected_image = np.ones(frame.shape)

    for i in range(frame.shape[0]):
        idx = np.where(mask[i,:] == 1)
        corrected_image[i, :] = frame[i,:] - np.nanmedian(frame[i,idx])
    return corrected_image

def gaussian(x, amp=1., mu=0., sig=1., offset=0.):
    """Gaussian function
    """
    exp = np.exp(-0.5 * ((x-mu)/sig)**2)
    return (amp*exp) + offset

def trace_spectrum(frame, xstart, xend, ystart, yend, kernel=None, radius=5, niters=100, **kwargs):
    """To find the trace of the spectrum in single frame
    
    This function finds the trace of the spectrum in single frame using deconvolution and centering.
    
    Parameters
    ----------
    frame : ndarray
        Single 2D frame image
    xstart : int
        Start column number of the trace
    xend : int
        End column number of the trace
    ystart : int
        Start row number for trace search
    yend : int
        End row number for trace search
    kernel : Callable function
        Callable function to define kernel
        Default is Gaussian function
    radius : int
        Radius of the aperture while defining the kernel
        Default is 5.
    niters : int
        Number of iteration for Richardson Lucy deconvolution
    **kwargs :
        Additional keywords provided to the `kernel` function
        
    Returns
    -------
    xpos : ndarray
        x-positions of the traced spectrum
    ypos : ndarray
        Traced position of the spectrum
    kern2D : ndarray
        2D kernel used in deconvolution
    deconv : ndarray
        Deconvolved image
    """
    data1 = np.copy(frame)
    # Removing <0 values and normalising the frame
    data1[data1<0] = 0
    data1 /= np.maximum(1000,np.max(data1, axis=0))[None,:]
    # Making kernel
    x1 = np.linspace(-radius,radius,2*radius + 1)
    if kernel is None:
        kern = gaussian(x=x1, **kwargs)
    else:
        kern = kernel(x=x1, **kwargs)
    ## Normalizing it
    kern = kern/np.max(kern)
    kern2D = np.reshape(kern, (len(kern), 1))
    deconv = restoration.richardson_lucy(data1, kern2D, num_iter=niters, clip=False)
    # Compute centre of flux
    row = np.arange(deconv.shape[0])
    centre = np.sum(deconv[ystart:yend,:]*row[ystart:yend, None], axis=0) / np.maximum(np.sum(deconv[ystart:yend,:], axis=0), 1)
    return np.arange(xstart, xend, 1, dtype=int), centre[xstart:xend], kern2D, deconv

def trace_spectra(frames, xstart, xend, ystart, yend, niter=3, clip=3, nknots=8, **kwargs):
    """Tracing spectra for multiple frames
    
    This function compute trace for all frames in input data using `trace_spectrum` function, and then fits a
    univariate spline to smooth the positions. "Master" median spline is then computed to define the shape robustly.
    Then for each frames the shape is assumed to be the same as the master median spline and fit for jitter.
    
    Parameters
    ----------
    frames : ndarray
        3D data cube with dimension, [nints, nrows, ncols]
    xstart : int
        Start column number of the trace
    xend : int
        End column number of the trace
    ystart : int
        Start row number for trace search
    yend : int
        End row number for trace search
    niter : int
        Number of iterations for various fitting procedure.
        Default is 3.
    clip : int
        Number of sigma clipping to mask points
        Default is 3.
    nknots : int
        Number of knots to be used in spline fitting
    **kwargs :
        Additional keywords provided to the `trace_spectrum` function
    
    Returns
    -------
    xpos : ndarray
        x-position of the traced spectra, same for each frame
    y_fitted_trace : ndarray
        Traced position of the spectra in each frame
    """
    # First find trace for each frame
    xpos, ycen2D = np.zeros(xend-xstart), np.zeros((frames.shape[0], xend-xstart))
    for i in tqdm(range(ycen2D.shape[0])):
        xpos, ycen2D[i,:], _, _ = trace_spectrum(frame=frames[i,:,:], xstart=xstart, xend=xend, ystart=ystart, yend=yend, **kwargs)
    
    # Median of the fitted spectrum
    medfl = medfilt2d(ycen2D, kernel_size=15)
    med_trace = np.median(medfl, axis=0)

    # Fitting spline to each of them
    knots = np.arange(xpos[0], xpos[-1], (xpos[-1] - xpos[0]) / np.double(nknots))[1:]
    y_interpolated = np.zeros(ycen2D.shape)
    ## Iterate through all integrations:
    for i in range(ycen2D.shape[0]):
        msk1 = np.ones(ycen2D.shape[1], dtype=bool)
        for _ in range(niter):
            if _ == 0:
                diff1 = ycen2D[i,:] - med_trace
            else:
                diff1 = ycen2D[i,:] - spl(xpos)
            limit = np.median(diff1[msk1]) + (clip*np.std(diff1[msk1]))
            msk1 = msk1 * (np.abs(diff1) < limit)
            spl = LSQUnivariateSpline(x=xpos[msk1], y=ycen2D[i,msk1], t=knots)
        y_interpolated[i,:] = spl(xpos)
    med_y_spline = np.nanmedian(y_interpolated, axis=0)   # Median fitted spline
    
    # Defining weights according to their difference w.r.t. median fitted spline
    weights = np.zeros(ycen2D.shape)
    for i in range(ycen2D.shape[0]):
        abs_diff = np.abs(ycen2D[i,:] - med_y_spline)
        diff_gr = np.where(abs_diff > 1.)[0]
        abs_diff[diff_gr] = 1.
        weights[i,:] = 1 - abs_diff
    
    # Again fitting spline to fitted trace, but now using weights
    y_interpolated_wt = np.zeros(ycen2D.shape)
    ## Iterate through all integrations:
    for i in range(ycen2D.shape[0]):
        msk2 = np.ones(ycen2D.shape[1], dtype=bool)
        for _ in range(niter):
            if _ == 0:
                diff2 = ycen2D[i,:] - med_y_spline
            else:
                diff2 = ycen2D[i,:] - spl(xpos)
            limit = np.median(diff2[msk2]) + (clip*np.std(diff2))
            msk2 = msk2 * (np.abs(diff1) < limit)
            spl = LSQUnivariateSpline(x=xpos[msk2], y=ycen2D[i,msk2], t=knots, w=weights[i,msk2])
        y_interpolated_wt[i,:] = spl(xpos)
    # Master median spline (which defines the shape of the spectrum)
    master_med_spline = np.nanmedian(y_interpolated_wt, axis=0)

    # Fitting for the jitter to this median fitted spline
    y_fitted_trace = np.zeros(ycen2D.shape)
    for i in range(ycen2D.shape[0]):
        msk3 = np.ones(ycen2D.shape[1], dtype=bool)
        for _ in range(3):
            if _ == 0:
                diff12 = ycen2D[i,:] - master_med_spline
            else:
                diff12 = ycen2D[i,:] - model_y_inter
            limit = np.median(diff12[msk3]) + (clip*np.std(diff12[msk3]))
            msk3 = msk3 * (np.abs(diff12) < limit)
            def residuals_all(xx):
                model = master_med_spline[msk3] + xx
                data = ycen2D[i,msk3]
                resids = np.sum(((data-model)*weights[i,msk3])**2)
                return resids
            soln_i_int = minimize(residuals_all, x0=0.01, method='BFGS')
            if np.abs(soln_i_int.x[0]) > 0.5:
                model_y_inter = master_med_spline
            else:
                model_y_inter = master_med_spline + soln_i_int.x[0]
        y_fitted_trace[i,:] = model_y_inter
    return xpos, y_fitted_trace