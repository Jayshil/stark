"""
Created on Thu Feb 16 16:36:09 2023

@author: Jayshil A. Patel (Initial version)
"""
import numpy as np
from skimage import restoration
from scipy.optimize import minimize
from scipy.interpolate import LSQUnivariateSpline
from tqdm import tqdm
import warnings
from scipy.signal import medfilt2d
from astropy.stats import SigmaClip, mad_std
try:
    from photutils.background import Background2D, MedianBackground, MMMBackground
except:
    print('`photutils` not installed!! Could not perforem 2D background subtraction.')

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
    sub_bkg : ndarray
        Array containing subtracted background from each column
    """
    corrected_image = np.ones(frame.shape)
    sub_bkg = np.ones(frame.shape[1])

    for i in range(frame.shape[1]):
        idx = np.where(mask[:,i] == 1)
        bkg1 = np.nanmedian(frame[idx,i])
        corrected_image[:, i] = frame[:,i] - bkg1
        sub_bkg[i] = bkg1
    return corrected_image, sub_bkg 

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
    sub_bkg : ndarray
        Array containing subtracted background from each row
    """
    corrected_image = np.ones(frame.shape)
    sub_bkg = np.ones(frame.shape[0])

    for i in range(frame.shape[0]):
        idx = np.where(mask[i,:] == 1)
        bkg1 = np.nanmedian(frame[i,idx])
        corrected_image[i, :] = frame[i,:] - bkg1
        sub_bkg[i] = bkg1
    return corrected_image, sub_bkg

def polynomial_bkg_cols(frame, mask, deg, sigma=5):
    """To fit a polynomial background subtraction along columns
    
    Given a frame, mask and degree of polynomial, this function fits a polynomial
    with specified degree to the unmasked background along columns.
    
    Parameters
    ----------
    frame : ndarray
        2D data frame of the data with shape [nrows, ncols]
    mask : ndarray
        Mask array with the same size as `frame`. Only points with 1 will be
        considered to estimate background.
    deg : int
        Degree of the polynomial
    sigma : int
        sigma clipping threshold for polynomial fitting
        Default is 5, use None in case of no sigma clipping
    
    Returns
    -------
    corrected_image : ndarray
        Background corrected image
    subtracted_bkg : ndarray
        Subtracted background, same shape as `corrected_image`"""
    
    corrected_image = np.copy(frame)
    subtracted_bkg = np.ones(frame.shape)

    for i in range(frame.shape[1]):
        try:
            idx_ok = np.where((mask[:,i] == 1)&(~np.isnan(frame[:,i])))[0]
            coeffs = np.polyfit(x=idx_ok, y=frame[idx_ok,i], deg=deg)
            poly = np.poly1d(coeffs)
            # Sigma clipping
            if sigma is not None:
                resids = frame[:,i] - poly(np.arange(frame.shape[0]))
                sigs = resids/mad_std(resids[idx_ok])
                idx_ok1 = idx_ok[sigs[idx_ok] < sigma]
                coeffs = np.polyfit(x=idx_ok1, y=frame[idx_ok1,i], deg=deg)
                poly = np.poly1d(coeffs)
            bkg1 = poly(np.arange(frame.shape[0]))
        except:
            bkg1 = np.zeros(frame.shape[0])
        # Subtracting background
        corrected_image[:,i] = corrected_image[:,i] - bkg1
        subtracted_bkg[:,i] = bkg1
    return corrected_image, subtracted_bkg

def polynomial_bkg_rows(frame, mask, deg, sigma=5):
    """To fit a polynomial background subtraction along rows
    
    Given a frame, mask and degree of polynomial, this function fits a polynomial
    with specified degree to the unmasked background along rows.

    
    Parameters
    ----------
    frame : ndarray
        2D data frame of the data with shape [nrows, ncols]
    mask : ndarray
        Mask array with the same size as `frame`. Only points with 1 will be
        considered to estimate background.
    deg : int
        Degree of the polynomial
    sigma : int
        sigma clipping threshold for polynomial fitting
        Default is 5, use None in case of no sigma clipping
    
    Returns
    -------
    corrected_image : ndarray
        Background corrected image
    subtracted_bkg : ndarray
        Subtracted background, same shape as `corrected_image`"""
    
    corrected_image = np.copy(frame)
    subtracted_bkg = np.ones(frame.shape)

    for i in range(frame.shape[0]):
        try:
            idx_ok = np.where((mask[i,:] == 1)&(~np.isnan(frame[i,:])))[0]
            coeffs = np.polyfit(x=idx_ok, y=frame[i,idx_ok], deg=deg)
            poly = np.poly1d(coeffs)
            # Sigma clipping
            if sigma is not None:
                resids = frame[i,:] - poly(np.arange(frame.shape[1]))
                sigs = resids/mad_std(resids[idx_ok])
                idx_ok1 = idx_ok[sigs[idx_ok] < sigma]
                coeffs = np.polyfit(x=idx_ok1, y=frame[idx_ok1,i], deg=deg)
                poly = np.poly1d(coeffs)
            bkg1 = poly(np.arange(frame.shape[0]))
        except:
            bkg1 = np.zeros(frame.shape[1])
        # Subtracting background
        corrected_image[i,:] = corrected_image[i,:] - bkg1
        subtracted_bkg[i,:] = bkg1
    return corrected_image, subtracted_bkg

def background2d(frame, mask, clip=5, bkg_estimator='median', box_size=(10,2)):
    """To perform 2D background subtraction

    Given the frame and mask this function will use `photutils` to estimate a 2D 
    background of the given image. Subsequently, it will subtract this background from the data.

    Parameters
    ----------
    frame : ndarray
        2D data frame of the data with shape [nrows, ncols]
    mask : ndarray
        Mask array with the same size as `frame`. Only points with 1 will be
        considered to estimate background.
    clip : int
        sigma clipping value
    bkg_estimator : str, either 'median' or 'mmm'
        See, bkg_estimator keyword in photutils.background.Background2D class for details
        Default is median
    box_size : (ny,nx)
        See, box_size keyword in photutils.background.Background2D class for details
        Default is (10,2)
    
    Returns
    -------
    corrected_image : ndarray
        Background corrected image
    bkg : ndarray
        Subtracted background"""

    data = np.copy(frame)
    mask2 = np.zeros(mask.shape, dtype=bool)
    # Because photutils.background.Background2D masks all points with True 
    # (not including them in computation, while in our masking scheme 1s are used in computations)
    mask2[mask == 0.] = True
    if bkg_estimator == 'median':
        bkg_est = MedianBackground()
    elif bkg_estimator == 'mmm':
        bkg_est = MMMBackground()
    else:
        raise Exception('Value of bkg_estimator can either be median or mmm.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', Warning)
        bkg = Background2D(data, box_size, mask=mask2, filter_size=(1,1), sigma_clip=SigmaClip(clip),\
                        bkg_estimator=bkg_est, fill_value=0.)
    return data - bkg.background, bkg.background

def gaussian(x, amp=1., mu=0., sig=1., offset=0.):
    """Gaussian function
    """
    exp = np.exp(-0.5 * ((x-mu)/sig)**2)
    return (amp*exp) + offset

def make_it_symmetric(arr):
    """Making array symmetric such that arr = np.flip(arr)"""
    mid_point = int(len(arr)/2)
    xargmax = np.argmax(arr)
    shift = mid_point - xargmax
    arr1 = np.roll(arr, shift)
    if shift < 0:
        arr2 = arr1[-1*shift:shift]
    elif shift == 0:
        arr2 = arr1
    else:
        arr2 = arr1[shift:-1*shift]
    arr3 = (arr2 + np.flip(arr2))/np.max(arr2 + np.flip(arr2))
    return arr3

def trace_spectrum(frame, xstart, xend, ystart, yend, kernel=None, radius=10, num_iter=100, **kwargs):
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
    num_iter : int
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
        kern1 = kernel(x=x1, **kwargs)
        kern = make_it_symmetric(kern1)
    ## Normalizing it
    kern = kern/np.max(kern)
    kern2D = np.reshape(kern, (len(kern), 1))
    deconv = restoration.richardson_lucy(data1, kern2D, num_iter=num_iter, clip=False)
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