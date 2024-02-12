Getting Started
===============

We will demonstrate the basic usage and idea behind the spectral extraction using :code:`stark`.
At the minimum, :code:`stark` requires a data and variance cube -- three dimensional arrays with
a shape of (nintegrations, nrows, ncolumns). Even if there is only one integration (frame) in the 
data, it needs to be converted to a 3D array with the mentioned dimensions. :code:`stark` further
assumes that the dispersion direction is along the rows. Other requirements for :code:`stark` is
the location of traces for each integration and aperture half-width. Optionally, a bad-pixel map and a 
normalisation vector (to normalise the data) can be provided. If not provided, then it is assumed
that none of the data points are "bad". See, the API documentation for more details.

Once all necessary products are gathered, the first task is to *load* the data using the 
:code:`SingleOrderPSF` class as follows:

.. code-block:: python

    from stark import SingleOrderPSF

    data = SingleOrderPSF(frame = data, variance = variance,\
                          ord_pos = trace_loc, ap_rad = aperture_half_width)


Executing above code would load the data in a pixel-table, which is a table created internally
by :code:`stark` to store and read all data. See, API documentation for how to read and access these data.

Once the data is loaded, we can fit splines to these data to obtain an estimate of stellar PSF.
There are three methods available to estimate the PSF.

- :code:`univariate_psf_frame`: this will fit a single 1D spline to *all* data as a function of pixel-coordinate
(i.e., distance from the trace). This can be achieved by running

.. code-block:: python
    
    psf_frame, psf_spline, mask_update = data.univariate_psf_frame()

A rule of thumb is that although this function is very good for determining an initial estimate of the PSF,
it will not produce a robust estimate of the PSF.

- :code:`univariate_multi_psf_frame`: this method will fit 1D splines to data within a moving window 
of N columns around a given column.

.. code-block:: python

    psf_frame, psf_spline, mask_update = data.univariate_multi_psf_frame(colrad = 50)

- :code:`bivariate_psf_frame`: this function will fit a 2D spline to the data as a function of
pixel-coordinate (distance from the trace) and wavelength (or, rather, column number).

.. code-block:: python

    psf_frame, psf_spline, mask_update = data.bivariate_psf_frame()

Each of above three methods have several additional keywords. The most important of them are
:code:`oversample` and :code:`knot_col`. The :code:`oversample` keyword determines how many knots 
to put in spatial direction while fitting a 1D spline. The default option is :code:`oversample = 1` which
will put 1 knot per pixel. Using :code:`oversample = 2` will double the number of knots and so on.
Similarly, :code:`knot_col` specifies the number of knots in the wavelength direction (only used 
when using :code:`bivariate_psf_frame`). The default is 10 knots in the wavelength/column direction, users
can change that number. Another noteworthy argument that each of three functions has is :code:`clip` keyword. While fitting
splines these functions can perform sigma clipping to mark outliers. The :code:`clip` argument 
specifies how many standard deviation away the data needs to be in order to identify as an outlier. 
The default is 5-sigma.

All three methods return 3 products: A :code:`psf_frame` (a 3D array with the same shape as data) gives the pixel sampled estimate of the 
stellar PSF, :code:`psf_spline` gives the best-fitted spline object that was used to generate
the :code:`psf_frame`, and updated mask (in :code:`mask_update`) after perfoming sigma clipping
during spline fitting. Note that the updated mask will be in the same format as it was in the "pixel table".
Users can use :code:`data.table2frame` function to convert it back to a 2D frame (see, API).

Once the PSF is estimated it is easy to compute the optimal extraction of the spectrum using:

.. code-block:: python

    from stark import optimal_extract

    spectrum, variance, synthetic_image = \
        optimal_extract(psf_frame = psf_frame[N,:,:], data = data[N,:,:],\
                        variance = variance[N,:,:], mask = mask[N,:,:],\
                        ord_pos = trace[N,:], ap_rad = aperture_half_width)

:code:`optimal_extract` accepts 2D arrays with (nrows, ncolumns) dimensions unlike :code:`SingleOrderPSF` class
which only accepts 3D arrays. This function will return an optimal estimation of the stellar 
spectrum, its variance and a synthetic data frame. A synthetic frame is a frame generated using
the best-fitted PSF (rather, pixel sampled estimate, :code:`psf_frame`) and optimal spectrum. One can
subtract this synthetic frame from the data frame to find residual frame which is an important diagnostic tool.

Additionally, :code:`stark` also provides some functions to perform a background subtraction and 
tracing the spectrum. Please see the API documentation for more details on this.