Pypam |version|
===============


Description
-----------

The goal of `pypam` is to allow easy reading and processing of acoustic underwater data.
`pypam` further depends on `pyhydrophone`_ for hydrophone metadata management and calibration.

`pypam` facilitates processing of audio files resulting from underwater acoustic deployments.
It enables application of existing methods of acoustic data processing, and it allows the processing of several
deployments with one line of code, so it is easy to create datasets to work with.
`pypam` is oriented to extracting features that can be used for machine learning algorithms or to the extraction of
broad acoustic information in time-series.

.. _pyhydrophone: https://github.com/lifewatch/pyhydrophone

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install
   quickstart

.. toctree::
   :caption: Documentation
   :maxdepth: 1

   available_features
   acoustic_file
   acoustic_survey
   dataset
   utils
   plots

.. toctree::
   :maxdepth: 2
   :caption: Example Gallery

   _auto_examples/index


Citing pypam
~~~~~~~~~~~~~~~~~~

.. note::
  If you find this package useful in your research, we would appreciate citations to:

Parcerisas (2023). lifewatch/pypam: A package to process underwater acoustics time series. Zenodo.
https://doi.org/10.5281/zenodo.5031689


About the project
~~~~~~~~~~~~~~~~~
This project has been funded by `LifeWatch Belgium <https://www.lifewatch.be/>`_.

.. image:: _static/lw_logo.png


For any questions please relate to clea.parcerisas@vliz.be


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
