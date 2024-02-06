# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Changed

- Row-by-row and column-by-column background subtraction function now also return the subtracted background.

### Added

- Functions to perform polynomial background subtraction along rows and columns.
- Function to perform a 2D background subtraction.

## [0.3.0]

### Changed

- Created a class for extracting psf frame for single order spectral data. This clss combines `flux_coo`, `norm_flux_coo`, `univariate_psf_frame` and `bivariate_psf_frame` functions.
- Changed the name of `psf_extract` to `optimal_extract` to avoid confusion.
- Updated docstrings
- Removed "oversample" in y (column) direction.

### Added

- Function to produce PSF using multi-column 1d spline fitting in `SingelOrderPSF` class.
- Functions to trace the spectrum in the data in `reduce.py`.

## [0.2.0] - 2023-02-24

### Added

- Added function to perform background subtraction.
- Created `reduce.py`
- Function to generate PSF frame using bivariate spline fitting to the whole dataset.
- Function to extract spectrum by fitting psf frame to the data.
- Function to generate psf frame using univariate spline fitting to the whole dataset.
- Functions to build a pixel array list and its normalized version.
- Univariate and bivariate spline fitting functions

## [0.1.0] - 2023-02-10

### Added

- Aperture extraction function in `extract.py`.
- CHANGELOG.md created