"""
Created on Thu Feb 16 16:36:09 2023

@author: Jayshil A. Patel (Initial version)
"""
import numpy as np

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