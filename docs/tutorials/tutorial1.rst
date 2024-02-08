Spectral extraction from a singe order transit timeseries data
==========

In the present notebook, we will extract spectra from a single order transit timeseries data. 
We will use WASP-39 transit timeseries data obtained with NIRCam/JWST for Transiting Exoplanet 
Community Early Release Science (ERS) program. All data products can be found on the 
`MAST portal <https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html>`_: this contains raw 
uncalibrated data (files with :code:`uncal.fits` extension), calibrated data (:code:`rateints.fits` 
and :code:`calints.fits`) and even spectrum timeseries data (:code:`x1dints.fits`). Here we will 
use :code:`rateints.fits` files to extract spectrum (see, the documentation of the 
`jwst <https://jwst-pipeline.readthedocs.io/en/latest/index.html>`_ pipeline to know more). 
We downloaded all files and put them in the same directory as this notebook.

We first need to "correct" this data for NaN values, 0s and cosmic rays, which will be our first task. 
We can then perform a background subtraction. Finally we will extract a spectrum timeseries from this 
dataset.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from stark import SingleOrderPSF, optimal_extract, reduce
    from astropy.stats import mad_std
    from glob import glob
    from astropy.io import fits
    from tqdm import tqdm
    from path import Path
    from scipy.optimize import curve_fit as cft
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import warnings

Loading the dataset
-------------------

Since the data volume is too big, the data products are delievered in segments. For WASP-39 NIRCam data, 
there are 4 segments. We first load data in all 4 segments and put them in a single numpy array below. The 
data in each segment contains several hundreads of individual exposures, or, frames. Each frame has a :code:`data` array, an :code:`error` array,
a :code:`dq` (data quality, i.e., bad-pixel map) array and a :code:`time` array. We will extract all 4 types
of information from each frame.

.. code-block:: python

    visit = 'NRCLW'

    # Input and Output paths
    p1 = os.getcwd()
    if not Path(p1 + '/Figures').exists():
        os.mkdir(p1 + '/Figures')

    ## Segments!!!
    segs = ['00' + str(i+1) for i in range(4)]

    # For 1st segment
    ## Loading the .fits file
    fhdul = glob(p1 + '/*seg001_nrcalong_rateints.fits')[0]
    hdul = fits.open(fhdul)
    ## 1, 2 and 3rd products are data, errors and bad-pixel map, respectively
    raw_data, raw_err, dq, time_bjd = hdul[1].data, hdul[2].data, hdul[3].data, hdul[4].data['int_mid_BJD_TDB']
    ## All values >0 in bad pixel maps are "bad"; we create a simpler bad-pixel map here,
    # 0 means bad pixel and 1 means a good pixel (the same convention used by stark)
    mask = np.ones(dq.shape)
    mask[dq > 0] = 0.

    # Repeating the same for the other segments
    for i in range(len(segs)-1):
        fhdul = glob(p1 + '/*seg' + segs[i+1] + '_nrcalong_rateints.fits')[0]
        hdul = fits.open(fhdul)
        # Data
        raw_data = np.vstack((raw_data, hdul[1].data))
        # Errors
        raw_err = np.vstack((raw_err, hdul[2].data))
        # DQ
        dq, m1 = hdul[3].data, np.ones(hdul[1].data.shape)
        m1[dq>0] = 0.
        mask = np.vstack((mask, m1))
        # Times
        time_bjd = np.hstack((time_bjd, hdul[4].data['int_mid_BJD_TDB']))

    time_bjd = time_bjd + 2400000.5
    nint = np.random.randint(0, raw_data.shape[0])


Correcting the dataset
----------------------

Although the data that we gathered above is a calibrated data, we still need to perform additional checks
to this dataset, looking for 0s and NaN, for instance. 0s and NaN values in error arrays will specially be 
painful since we aim to use error array as weighting while fitting a PSF. So, let's first correct for 0s and
NaN from the error array. We will, additionally, consider these pixels as "bad" and add them to the default
bad-pixel map.

.. code-block:: python

    ## Correct errorbars
    print('>>>> --- Correcting errorbars (for zeros and NaNs)...')
    med_err = np.nanmedian(raw_err.flatten())
    ## Changing Nan's and zeros in error array with median error
    corr_err1 = np.copy(raw_err)
    corr_err2 = np.where(raw_err != 0., corr_err1, med_err)
    corrected_errs = np.where(np.isnan(raw_err) != True, corr_err2, med_err)
    print('>>>> --- Done!!')

    print('>>>> --- Updating the bad-pixel map...')
    ## Making a bad-pixel map (1s are good pixels, 0s are bad pixels)
    mask_bp1 = np.ones(raw_data.shape)
    mask_bp2 = np.where(raw_err != 0., mask_bp1, 0.)               # This will place 0 in mask where errorbar == 0
    mask_bp3 = np.where(np.isnan(raw_err) != True, mask_bp2, 0.)   # This will place 0 in mask where errorbar is Nan
    mask_badpix = mask * mask_bp3     # Adding those pixels which are identified as bad by the pipeline (and hence 0)
    print('>>>> --- Done!!')

.. code-block:: bash

    >>>> --- Correcting errorbars (for zeros and NaNs)...
    >>>> --- Done!!
    >>>> --- Updating the bad-pixel map...
    >>>> --- Done!!

Our data will be contaminated with a lot of cosmic rays, we want to identify those pixels and add them 
to our bad pixel map. Our method of identifying cosmic rays is pretty simple: we will generate a median 
dataframe, and compare this median frame with all frames. Since cosmic rays are outliers, we should be 
able to identify them by comparing each frame with a median frame. We further want to correct these 
values by taking mean of neighbouring pixels. 

.. code-block:: python

    def identify_crays(frames, mask_bp, clip=5, niters=5):
        """Given a data cube and bad-pixel map, this function identifies cosmic rays by using median frame"""
        # Masking bad pixels as NaN
        mask_cr = np.copy(mask_bp)
        for _ in range(niters):
            # Flagging bad data as Nan
            frame_new = np.copy(frames)
            frame_new[mask_cr == 0.] = np.nan
            # Median frame
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median_frame = np.nanmedian(frame_new, axis=0)  # 2D frame
                # Creating residuals
                resids = frame_new - median_frame[None,:,:]
                # Median and std of residuals
                med_resid, std_resid = np.nanmedian(resids, axis=0), np.nanstd(resids, axis=0)
            limit = med_resid + (clip*std_resid)
            mask_cr1 = np.abs(resids) < limit[None,:,:]
            mask_cr = mask_cr1*mask_bp
        return mask_cr

    def replace_nan(data, max_iter = 50):
        """Replaces NaN-entries by mean of neighbours.
        Iterates until all NaN-entries are replaced or
        max_iter is reached. Works on N-dimensional arrays.
        """
        nan_data = data.copy()
        shape = np.append([2*data.ndim], data.shape)
        interp_cube = np.zeros(shape)
        axis = tuple(range(data.ndim))
        shift0 = np.zeros(data.ndim, int)
        shift0[0] = 1
        shift = []     # Shift list will be [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for n in range(data.ndim):
            shift.append(tuple(np.roll(-shift0, n)))
            shift.append(tuple(np.roll(shift0, n)))
        for _j in range(max_iter):
            for n in range(2*data.ndim):
                interp_cube[n] = np.roll(nan_data, shift[n], axis = axis)   # interp_cube would be (4, data.shape[0], data.shape[1]) sized array
            with warnings.catch_warnings():                                 # with shifted position in each element (so that we can take its mean)
                warnings.simplefilter('ignore', RuntimeWarning)
                mean_data = np.nanmean(interp_cube, axis=0)
            nan_data[np.isnan(nan_data)] = mean_data[np.isnan(nan_data)]
            if np.sum(np.isnan(nan_data)) == 0:
                break
        return nan_data

    ## Mask with cosmic rays
    ### Essentially this mask will add 0s in the places of bad pixels...
    print('>>>> --- Identifying cosmic rays and updating the bad-pixel map...')
    mask_bcr = identify_crays(raw_data, mask_badpix)
    print('Total per cent of masked points:\
        {:.4f} %'.format(100 * (1 - np.sum(mask_bcr) / (mask_bcr.shape[0] * mask_bcr.shape[1] * mask_bcr.shape[2]))))
    print('>>>> --- Done!!')

    # And interpolating the data in bad-pixels with mean of neighbouring pixels
    print('>>>> --- Correcting data...')
    corrected_data_wo_bkg = np.copy(raw_data)
    corrected_data_wo_bkg[mask_bcr == 0] = np.nan
    for i in range(corrected_data_wo_bkg.shape[0]):
        corrected_data_wo_bkg[i,:,:] = replace_nan(corrected_data_wo_bkg[i,:,:])
    print('>>>> --- Done!!')

.. code-block:: bash

    >>>> --- Identifying cosmic rays and updating the bad-pixel map...
    Total per cent of masked points:      5.4090 %
    >>>> --- Done!!
    >>>> --- Correcting data...
    >>>> --- Done!!

Let's now visualise our data -- we will display one randomly selected frame below:

.. code-block:: python

    plt.figure(figsize=(15,5))
    plt.imshow(corrected_data_wo_bkg[nint,4:,:], interpolation='none', aspect='auto')
    plt.title('Example data frame')

.. figure:: Example_data_frame.png
   :alt: Example data frame

It looks good! So, there are 256 rows (spatial direction) and 2048 columns (dispersion direction). 
The location of the trace is clearly seen. Let's plot the value of flux for a given column: