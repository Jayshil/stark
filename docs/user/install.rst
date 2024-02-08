Installation
============

The easiest (and recommended) method to install :code:`stark` is by using pip as the following:

.. _pip_install:

pip installation
+++++++++++

.. code-block:: bash

    pip install stark-package

.. _source_install:

Installing from source
+++++++++++

Alternatively, the users can download the latest developement version of :code:`stark` directly from 
the cource by doing:

.. code-block:: bash

    git clone https://github.com/Jayshil/stark.git
    cd stark
    python setup.py install

.. _dependencies:

Dependencies
+++++++++++

:code:`stark` has some very basic dependencies on :code:`numpy`, :code:`scipy` and :code:`astropy`, 
which should already be installed if you are using an Anaconda installation. Furthermore, :code:`stark` 
uses :code:`photutils <https://photutils.readthedocs.io/en/stable/>_` (which is an optional dependency) 
package to estimate 2D background. Not installing :code:`photutils` will mute this ability of :code:`stark`.