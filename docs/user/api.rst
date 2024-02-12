API
===
.. module:: stark

The simplest way to extract spectrum is performing aperture extraction. While not robust, it can provide an initial guess of the spectrum.
With :code:`stark`, it is possible to do aperture extraction of spectrum using the following function.

.. autofunction:: stark.aperture_extract

The main class of :code:`stark` is :code:`SingleOrderPSF` which has functionalities to read single order PSF data whether it is timeseries data or not.
Users can first load the data using this class, and then use its different methods to fit a univariate or bivariate spline to it to estimate the stellar PSF.
Details of this class and its methods are as below:

.. autoclass:: stark.SingleOrderPSF
   :members:

An optimal extraction of the stellar spectrum can be computed with a robust estimate of stellar PSF found from :code:`SingleOrderPSF` class.

.. autofunction:: stark.optimal_extract

The three methods to estimate PSF used by the :code:`SingleOrderPSF` class, in turn, uses the 
following functions to fit splines to the data:

.. autofunction:: stark.fit_spline_univariate

.. autofunction:: stark.fit_spline_bivariate

Apart from spectral extraction, :code:`stark` also provides tools for background subtraction and tracing the spectrum:

.. autofunction:: stark.reduce.col_by_col_bkg_sub

.. autofunction:: stark.reduce.row_by_row_bkg_su

.. autofunction:: stark.reduce.polynomial_bkg_cols

.. autofunction:: stark.reduce.polynomial_bkg_rows

.. autofunction:: stark.reduce.background2d

.. autofunction:: stark.reduce.trace_spectrum

.. autofunction:: stark.reduce.trace_spectra

(Last two functions are computationally expensive.)