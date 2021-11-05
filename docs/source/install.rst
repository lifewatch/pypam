.. currentmodule:: pypam

Install
=======

We strongly advise to use a proper python virtualenv environment to keep package dependencies nicely non conflicting
Please read `here <https://docs.python.org/3/tutorial/venv.html>`_ if you don't know about it!
You will never stop using it anymore.


Using pip distribution
----------------------
.. code-block::

    pip install lifewatch-pypam

Using git clone
---------------

1. Download the code using git (can be substituted by downloading the zip folder of the code)

.. code-block::

    git clone https://github.com/lifewatch/pypam.git


2. Use the package manager `pip <https://pip.pypa.io/en/stable/>`_ to install the remaining dependencies. The package
dependency pyhydrophone should be automatically downloaded and installed.

.. code-block::

    pip install -r requirements.txt


3. Build the project

.. code-block::

    python setup.py install
