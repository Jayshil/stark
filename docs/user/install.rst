Installation
============

The easiest (and recommended) method to install :code:`stark` is by using pip as the following:

.. code-block:: bash

    pip install stark-package

Alternatively, the users can download the latest developement version of :code:`stark` directly from 
the cource by doing:

.. code-block:: bash

    git clone https://github.com/Jayshil/stark.git
    cd stark
    python setup.py install

Dependencies
+++++++++++

:code:`stark` has some very basic dependencies on :code:`numpy`, :code:`scipy` and :code:`astropy`, 
which should already be installed if you are using an Anaconda installation. Furthermore, :code:`stark` 
uses `photutils <https://photutils.readthedocs.io/en/stable/>`_ (which is an optional dependency) 
package to estimate 2D background used in some background subtraction methods implemented in :code:`stark`.