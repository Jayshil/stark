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
from tqdm import tqdm

def aperture_extract(frame, variance, ord_pos, ap_rad, uniform = False):
    """ Simple aperture extraction of the spectrum

    Given the 2D `frame` of the data, and its `variance`, this function extracts spectrum by simply
    adding up values of pixels along slit.

    Parameters
    ----------
    frame : ndarray
        2D data frame from which the spectrum is to be extracted
    variance : ndarray
        The noise image of the same format as frame, specifying the variance of each pixel.
    ord_pos : ndarray
        Array defining position of the trace
    ap_rad : float
        Radius of the aperture around the order position
    uniform : bool, optional
        Boolean on whether the slit is uniformally lit or not.
        If not then it will simply sum up counts in the aperture, else average the counts and multiply by slit-length.
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
        i1 = int(round(ord_pos[col] + ap_rad)) + 1

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

class SingleOrderPSF(object):
    """
    Given a data frame, its variance, order positions and aperture radius, this class generates a PSF frame for the data. The input
    data should be a 3D data cube with dimensions [nints, nrows, ncols], with variace array of the same shape. Optionally,
    a normalisation spectra and bad pixel map can be provided. This class works only for single order spectrum.

    Example usage:
    >>> data = stark.SingleOrderPSF(frame=data_frame, variance=data_variance, ord_pos=ord_pos, ap_rad=aperture_radius)
    >>> psf_frame, psf_spline = data.univariate_psf_frame()

    when fitting a univariate spline, as function of pixel coordinate, to the whole dataset, or,

    >>> data = stark.SingleOrderPSF(frame=data_frame, variance=data_variance, ord_pos=ord_pos, ap_rad=aperture_radius)
    >>> psf_frame, psf_spline = data.bivariate_psf_frame()

    when fitting bivariate spline, as function of pixel coordinate and column number, to the whole dataset.

    Here, `psf_frame` is resultant PSF frame and `psf_spline` is `scipy.interpolate.LSQUnivariateSpline` (or, `scipy.interpolate.LSQBivariateSpline`) object.

    Parameters
    ----------
    frame : ndarray
        3D array containing data, [nints, nrows, ncols]
    variance : ndarray
        3D array as same size as `frame` containing variance of the data.
    ord_pos : ndarray
        2D array, with shape [nints, ncols] contaning pixel positions of order
    ap_rad : float
        Aperture radius
    spec : ndarray, optional
        2D array, of [nints, ncols] size, providing normalisation spectrum.
    mask : ndarray, optional
        3D array containing mask, same shape as `frame`.
        Default is None.
    """
    def __init__(self, frame, variance, ord_pos, ap_rad, spec=None, mask=None):
        self.frame = frame
        self.variance = variance
        self.ord_pos = ord_pos
        self.ap_rad = ap_rad
        self.spec = spec

        if mask is None:
            self.mask = np.ones(self.frame.shape)
        else:
            self.mask = mask

        # Data
        self.nints = frame.shape[0]
        self.ncols = frame.shape[2]

        # Creating pixel list
        self.flux_coo()
        # Generating normalized pixel array
        self.norm_flux_coo()
    
    def flux_coo(self):
        """ To produce pixel list with source coordinates
        
        Given a 3D data cube, this function produces an array, stored as `self.pix_array`, with pixel coordinates (distace from order position), 
        their flux, variances, column number and mask. Also generates a 3D array (`self.col_array_pos`) of size [nints, ncols, 2] to store positions
        of pixels in `self.pix_array`. E.g., if the data in `self.pix_array` from columns 100 to 300 are desired, those corresponds to indices
        `self.col_array_pos[i,100,0] to `self.col_array_pos[i,300,0]` for i-th integration in the `self.pix_array`. 
        This information is to be used when producing PSFs. This function works for single order spectrum.

        """
        
        self.col_array_pos = np.zeros((self.nints, self.ncols, 2), dtype=int) # position and length in array
        pix_list = []
        col_pos = 0

        for integration in range(self.nints):
            for col in range(self.ncols):
                self.col_array_pos[integration, col, 0] = col_pos

                if self.ord_pos[integration, col] < 0 or self.ord_pos[integration, col] >= self.frame.shape[1]:
                    continue
                i0 = int(round(self.ord_pos[integration, col] - self.ap_rad))
                i1 = int(round(self.ord_pos[integration, col] + self.ap_rad)) + 1

                if i0 < 0:
                    i0 = 0
                if i1 >= self.frame.shape[1]:
                    i1 = self.frame.shape[1] - 1
                npix = i1-i0                       # Length of aperture
                col_array = np.zeros((npix,5))     # (aper_size, 3) array, containing,
                col_array[:,0] = np.array(range(i0,i1))-self.ord_pos[integration, col]    # pix position from center
                col_array[:,1] = self.frame[integration, i0:i1, col]                      # data at those points, and
                col_array[:,2] = self.variance[integration, i0:i1, col]                   # variance on those data points
                col_array[:,3] = np.ones(npix)*col                                        # Column number
                col_array[:,4] = self.mask[integration, i0:i1, col]                       # mask
                self.col_array_pos[integration, col, 1] = npix
                col_pos += npix
                pix_list.append(col_array)         # Is a list containing col_array for each column
            
        # Make continuous array out of list of arrays
        num_entries = np.sum([p.shape[0] for p in pix_list])
        self.pix_array = np.zeros((num_entries,5))
        entry = 0
        for p in pix_list:
            N = len(p)
            self.pix_array[entry:(entry+N),:] = p
            entry += N

    def norm_flux_coo(self):
        """ Normalises the fluxes by summing up pixel values.
        
        Given the `self.pixel array` and `self.col_array_pos` from `self.flux_coo` function, this function provides the normalized fluxes.
        If no normalisation spectrum is provided, the pixel sum is used.
        
        """
        
        # Initializing an array for normalized pixel list
        self.norm_array = self.pix_array.copy()
        
        min_norm = 0.01
        for integration in range(self.nints):
            for col in range(self.ncols):
                ind0 = self.col_array_pos[integration, col, 0]
                ind1 = ind0 + self.col_array_pos[integration, col, 1]
                if self.spec is None:
                    norm_sum = np.sum(self.pix_array[ind0:ind1,1])
                else:
                    norm_sum = self.spec[integration, col]
                norm_sum = np.maximum(norm_sum, min_norm)
                self.norm_array[ind0:ind1,1] = self.pix_array[ind0:ind1,1]/norm_sum
                self.norm_array[ind0:ind1,2] = self.pix_array[ind0:ind1,2]/norm_sum**2

    def univariate_psf_frame(self, **kwargs):
        """To generate PSF frame by fitting univariate spline for the whole dataset
        
        Given a `stark.SingleOrderPSF` object, this function derives a single PSF for whole data cube and use it to 
        generate pixel-sampled PSFs for each column.

        Parameters
        ----------
        **kwargs :
            Additional keywords provided to `fit_spline_univariate` function and to LSQUnivariateSpline
        
        Returns
        -------
        psf_frame : ndarray
            Data cube containing pixel-sampled PSF for each column
        psf_spline : `scipy.interpolate.LSQUnivariateSpline`
            Fitted spline object
        """
        
        psf_frame = np.copy(self.frame)

        # To sort array
        sortarg =  np.argsort(self.norm_array[:,0])
        sort_arr = self.norm_array[sortarg,:]
        psf_spline, mask = fit_spline_univariate(sort_arr, **kwargs)

        for integration in range(self.nints):
            for col in range(self.ncols):
                if self.ord_pos[integration, col] < 0 or self.ord_pos[integration, col] >= psf_frame.shape[1]:
                    continue
                i0 =  int(round(self.ord_pos[integration, col] - self.ap_rad))
                i1 = int(round(self.ord_pos[integration, col] + self.ap_rad)) + 1
                if i0 < 0:
                    i0 = 0
                if i1 >= psf_frame.shape[1]:
                    i1 = psf_frame.shape[1] - 1

                x = np.arange(i0,i1) - self.ord_pos[integration, col]
                psf_frame[integration, i0:i1, col] = np.maximum(psf_spline(x), 0) # Enforce positivity
                psf_frame[integration, i0:i1, col] /= np.sum(psf_frame[integration, i0:i1, col]) # Enforce normalisation (why commented)
        return psf_frame, psf_spline
    
    def univariate_multi_psf_frame(self, colrad = 100, **kwargs):
        """ To generate PSF frame using fitting 1D-spline to a window of multiple columns around given column
        
        To derive one PSF spline for each column, and uses it to generate the pixel-sampled
        PSF for each column.

        Parameters
        ----------
        colrad : int
            Radius of column window
            Default is 100
        **kwargs :
            Additional keywords provided to `fit_spline_univariate` function and to LSQUniivariateSpline
        """

        nretries = 4
        curr_rad = colrad

        psf_frame = np.copy(self.frame)
        psf_spline_all = []
        
        for col in tqdm(range(self.ncols)):
            for retry in range(nretries):
                try:        
                    col_start = int(round(col - curr_rad))
                    col_end = int(round(col + curr_rad))
                    if col_start < 0:
                        col_start = 0
                    if col_end >= self.ncols:
                        col_end = self.ncols-1
                    
                    # Indices of the sub-array
                    ind1 = np.where((self.norm_array[:,3]>=col_start)&(self.norm_array[:,3]<=col_end))
                    sub_arr = self.norm_array[ind1[0],:]               # Desired sub-array
                    sort_arr = sub_arr[sub_arr[:,0].argsort(),:]       # Sorted sub-array for 1d spline fitting
                    psf_spline, mask = fit_spline_univariate(sort_arr, **kwargs)
                    psf_spline_all.append(psf_spline)
                except IndexError:
                    curr_rad *= 2
                    print('Retrying with colrad = {:d} columns'.format(curr_rad))
                    continue
                break
            curr_rad = colrad

            for integration in range(self.nints):
                if self.ord_pos[integration, col] < 0 or self.ord_pos[integration, col] >= psf_frame.shape[1]:
                    continue
                i0 =  int(round(self.ord_pos[integration, col] - self.ap_rad))
                i1 = int(round(self.ord_pos[integration, col] + self.ap_rad)) + 1
                if i0 < 0:
                    i0 = 0
                if i1 >= psf_frame.shape[1]:
                    i1 = psf_frame.shape[1] - 1
                
                x = np.arange(i0,i1) - self.ord_pos[integration, col]
                psf_frame[integration, i0:i1, col] = np.maximum(psf_spline(x), 0) # Enforce positivity
                psf_frame[integration, i0:i1, col] /= np.sum(psf_frame[integration, i0:i1, col]) # Enforce normalisation

        return psf_frame, psf_spline_all

    def bivariate_psf_frame(self, **kwargs):
        """To generate PSF frame by fitting a bivariate spline to the whole dataset
        
        Given a `stark.SingleOrderPSF` object, this function derives a single 2D PSF for whole data cube and use it to 
        generate pixel-sampled PSFs for each column.

        Parameters
        ----------
        **kwargs :
            Additional keywords provided to `fit_spline_bivariate` function and to LSQBivariateSpline
        
        Returns
        -------
        psf_frame : ndarray
            Data cube containing pixel-sampled PSF for each column
        psf_spline : `scipy.interpolate.LSQUnivariateSpline`
            Fitted spline object
        """
        
        psf_frame = np.copy(self.frame)

        # To sort array
        psf_spline, mask = fit_spline_bivariate(self.norm_array, **kwargs)

        for integration in range(self.nints):
            for col in range(self.ncols):
                if self.ord_pos[integration, col] < 0 or self.ord_pos[integration, col] >= psf_frame.shape[1]:
                    continue
                i0 =  int(round(self.ord_pos[integration, col] - self.ap_rad))
                i1 = int(round(self.ord_pos[integration, col] + self.ap_rad)) + 1
                if i0 < 0:
                    i0 = 0
                if i1 >= psf_frame.shape[1]:
                    i1 = psf_frame.shape[1] - 1

                x = np.arange(i0,i1) - self.ord_pos[integration, col]
                psf_frame[integration, i0:i1, col] = np.maximum(psf_spline(x, np.ones(x.shape)*col, grid=False), 0) # Enforce positivity
                psf_frame[integration, i0:i1, col] /= np.sum(psf_frame[integration, i0:i1, col])                        # Enforce normalisation
        return psf_frame, psf_spline

def optimal_extract(psf_frame, data, variance, mask, ord_pos, ap_rad):
    """Use derived PSF frame (the psf sampled on the image) to extract the spectrum.

    This function fits the derived PSF frame to the actual data to extract the
    spectrum and variance on it.

    Parameters
    ----------
    psf_frame : ndarray
        PSF frame with dimension [nrows, ncols]
    data : ndarray
        2D array containing data, [nrows, ncols]
    variance : ndarray
        Variance on the data frame, same shape as the `data` array
    mask : ndarray
        Array containing mask; only those points with value = True will be considered in extraction
    ord_pos : ndarray
        1D array, with length ncols, contaning pixel positions of order
    ap_rad : float
        Radius of the aperture to consider
    
    Returns
    -------
    spec : ndarray
        Extracted flux as a matrix with format [ncols]
    var : ndarray
        Variance of each fluc point in the same format as `spec`.
    synth : ndarray
        Synthetic image constructed using the flux and the PSF frame;
        useful for producing a residual image.
    """

    ncols = data.shape[1]

    spec = np.zeros(ncols)
    var = np.zeros(ncols)
    synth = np.zeros(data.shape)

    for col in range(ncols):
        if ord_pos[col] < 0 or ord_pos[col] >= data.shape[0]:
            continue
        i0 =  int(round(ord_pos[col] - ap_rad))
        i1 = int(round(ord_pos[col] + ap_rad)) + 1

        if i0 < 0:
            i0 = 0
        if i1 >= data.shape[0]:
            i1 = data.shape[0] - 1

        ind = np.array(range(i0,i1))
        mask2 = psf_frame[ind, col] > 0
        ind = ind[mask2]
        denom = np.sum(mask[ind, col] * psf_frame[ind, col]**2 / 
                        variance[ind, col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec[col] = (np.sum(mask[ind, col] * psf_frame[ind, col] *
                data[ind, col] / variance[ind, col]) / denom )
            var[col] = np.sum(mask[ind, col] * psf_frame[ind, col]) / denom
        synth[i0:i1, col] = spec[col] * psf_frame[i0:i1, col]
    return spec, var, synth

def fit_spline_univariate(pixel_sorted, oversample=1, clip=5, niters=3, verbose=True, **kwargs):
    """Fit a Univariate spline to pixel arrays
    
    Given a normalized sorted pixel array (in the same form as output from 
    `norm_flux_coo`) this function fit univariate spline as a function of
    pixel coordinates.

    Parameters
    ----------
    pixel_sorted : ndarray
        Normalized pixel array with pixel coordinates, normalized flux, normalized
        variance, and column indices; sorted according to pixel coordinates
    oversample : int, optional
        Determines the number of knots in pixel coordindates direction. Default number
        of knots is equal to total pixel numbers in aperture (corresponds to `oversample`=1).
        If `oversample` is greater than 1, it will put oversamples*total pixel number knots
        in pixel coordinate direction.
    clip : int, optinal
        Number of sigmas to perform sigma clipping while fitting a spline.
        Default is 5.
    niter : int, optional
        Number of iteration to perform
        Default is 3.
    verbose : bool, optional
        Boolean on whether to print details or not.
        Default is True.
    **kwargs :
        Additional keywords provided to LSQUnivariateSpline
    
    Returns
    -------
    psf_spline : scipy.interpolate.LSQUnivariateSpline object
        Fitted spline object.
    mask : ndarray
        Array containing location of masked points.
    """
    t0 = np.min(pixel_sorted[:,0]) + 1/oversample
    t1 = np.max(pixel_sorted[:,0]) - 1/oversample
    t = np.linspace(t0, t1, int(t1-t0)*int(oversample))

    # Mask for bad pixels
    mask_bp = np.asarray(pixel_sorted[:,4], dtype=bool)

    weights = np.maximum(1/pixel_sorted[:,2], 0)
    psf_spline = LSQUnivariateSpline(x=pixel_sorted[mask_bp,0], 
                                     y=pixel_sorted[mask_bp,1],
                                     t=t,
                                     w=weights[mask_bp])
    
    mask = np.ones(len(pixel_sorted[:,0]), dtype=bool) * mask_bp
    for i in range(niters):
        # Sigma clipping
        resids = pixel_sorted[:,1] - psf_spline(pixel_sorted[:,0])
        limit = np.median(resids[mask]) + (clip*np.std(resids[mask]))
        mask = np.abs(resids) < limit
        mask = mask * mask_bp
        # And spline fitting
        psf_spline = LSQUnivariateSpline(x=pixel_sorted[mask,0], 
                                 y=pixel_sorted[mask,1],
                                 t=t,
                                 w=weights[mask])
        
        if verbose:
            print('Iter {:d} / {:d}: {:.5f} per cent masked.'.format(i+1, niters, 100 - 100*np.sum(mask)/len(pixel_sorted[:,0])))
    return psf_spline, mask

def fit_spline_bivariate(pixel_array, oversample=1, knot_col=10, clip=5, niters=3, verbose=True, **kwargs):
    """Fit a Bivariate spline to pixel arrays
    
    Given a normalized pixel array (in the same form as output from 
    `norm_flux_coo`) this function fit bi-variate spline as a function of
    pixel coordinates and column indices.

    Parameters
    ----------
    pixel_array : ndarray
        Normalized pixel array with pixel coordinates, normalized flux, normalized
        variance, and column indices.
    oversample : int, optional
        Determines the number of knots in pixel coordindates direction. Default number
        of knots is equal to total pixel numbers in aperture (corresponds to `oversample`=1).
        If `oversample` is greater than 1, it will put oversamples*total pixel number knots
        in pixel coordinate direction.
    knot_col : int, optional
        Number of knots in column indices direction.
        Default is 10.
    clip : int, optinal
        Number of sigmas to perform sigma clipping while fitting a spline.
        Default is 5.
    niter : int, optional
        Number of iteration to perform
        Default is 3.
    verbose : bool, optional
        Boolean on whether to print details or not.
        Default is True.
    **kwargs :
        Additional keywords provided to LSQBivariateSpline
    
    Returns
    -------
    psf_spline : scipy.interpolate.LSQBivariateSpline object
        Fitted spline object.
    mask : ndarray
        Array containing location of masked points.
    """
    x1k, x2k = np.min(pixel_array[:,0]) + 1/oversample, np.max(pixel_array[:,0]) - 1/oversample
    y1k, y2k = np.min(pixel_array[:,3]) + 1, np.max(pixel_array[:,3]) - 1

    xknots = np.linspace(x1k, x2k, int(x2k-x1k)*int(oversample))
    yknots = np.linspace(y1k, y2k, int(knot_col))

    # Mask for bad pixels
    mask_bp = np.asarray(pixel_array[:,4], dtype=bool)

    weights = np.maximum(1/pixel_array[:,2], 0)
    psf_spline = LSQBivariateSpline(x=pixel_array[mask_bp,0], y=pixel_array[mask_bp,3],\
        z=pixel_array[mask_bp,1],\
        tx=xknots, ty=yknots,\
        w=weights[mask_bp], **kwargs)
    
    mask = np.ones(len(pixel_array[:,0]), dtype=bool) * mask_bp
    for i in range(niters):
        # Sigma clipping
        resids = pixel_array[:,1] - psf_spline(pixel_array[:,0], pixel_array[:,3], grid=False)
        limit = np.median(resids[mask]) + (clip*np.std(resids[mask]))
        mask = np.abs(resids) < limit
        mask = mask * mask_bp
        # And spline fitting
        psf_spline = LSQBivariateSpline(x=pixel_array[mask,0], y=pixel_array[mask,3],\
            z=pixel_array[mask,1],\
            tx=xknots, ty=yknots,\
            w=weights[mask], **kwargs)
        
        if verbose:
            print('Iter {:d} / {:d}: {:.5f} per cent masked.'.format(i+1, niters, 100 - 100*np.sum(mask)/len(pixel_array[:,0])))
    return psf_spline, mask