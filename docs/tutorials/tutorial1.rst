Spectral extraction from a singe order transit timeseries data
==========

In the present notebook, we will extract spectra from a single order transit timeseries data. We will use WASP-39 transit timeseries data obtained with NIRCam/JWST for Transiting Exoplanet Community Early Release Science (ERS) program. All data products can be found on the `MAST portal <https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html>`_: this contains raw uncalibrated data (files with :code:`uncal.fits` extension), calibrated data (:code:`rateints.fits` and :code:`calints.fits`) and even spectrum timeseries data (:code:`x1dints.fits`). Here we will use :code:`rateints.fits` files to extract spectrum (see, the documentation of the `jwst <https://jwst-pipeline.readthedocs.io/en/latest/index.html>`_ pipeline to know more). We downloaded all files and put them in the same directory as this notebook.

We first need to "correct" this data for NaN values, 0s and cosmic rays, which will be our first task. We can then perform a background subtraction. Finally we will extract a spectrum timeseries from this dataset.

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

Since the data volume is too big, the data products are delievered in segments. For WASP-39 NIRCam data, there are 4 segments. We first load data in all 4 segments and put them in a single numpy array below:

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