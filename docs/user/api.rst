API
===
.. module:: stark

The simplest way to extract spectrum is performing aperture extraction. While not robust, it can provide an initial guess of the spectrum.
With `stark`, it is possible to do aperture extraction of spectrum using the following function.

.. autofunction:: stark.aperture_extract

The main class of `stark` is `SingleOrderPSF` which has functionalities to read single order PSF data whether it is timeseries data or not.
Users can first load the data using this class, and then use its different methods to fit a univariate or bivariate spline to it to estimate the stellar PSF.
Details of this class and its methods are as below:

.. autoclass:: stark.SingleOrderPSF
   :members:

An optimal extraction of the stellar spectrum can be computed with a robust estimate of stellar PSF found from `SingleOrderPSF` class.

.. autofunction:: stark.optimal_extract