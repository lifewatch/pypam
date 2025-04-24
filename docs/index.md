# *Pypam*: a package to process underwater acoustics time series

!!! note "WIP"
    Thanks for your interest in pypam. The package and the documentation are still a work in progress :construction:.
    Please get in touch if you have any questions or suggestions.

## Description

The goal of *pypam* is to allow easy reading and processing
of acoustic underwater data. *pypam* further depends on
[pyhydrophone](https://github.com/lifewatch/pyhydrophone) for hydrophone
metadata management and calibration.

*pypam* facilitates processing of audio files resulting from
underwater acoustic deployments. It enables application of existing
methods of acoustic data processing, and it allows the processing of
several deployments with one line of code, so it is easy to create
datasets to work with. *pypam* is oriented to extracting
features that can be used for machine learning algorithms or to the
extraction of broad acoustic information in time-series.

*pypam* works with [xarray](https://xarray.dev/) structures as output, making it very easy to store the results 
using different formats (we recommend netCDF). 

![image](./source/_static/PyPAM_colour_white_bg.png)
Logo by [Dr. Stan Pannier](https://www.vliz.be/en/imis?module=person&persid=37468)


### Citing pypam

If you find this package useful in your research, please cite:

Parcerisas (2023). lifewatch/pypam: A package to process underwater
acoustics time series. Zenodo. <https://doi.org/10.5281/zenodo.5031689>

### About the project

This project has been funded by [LifeWatch Belgium](https://www.lifewatch.be/).

![image](./source/_static/lw_logo.png)

For any questions please relate to <clea.parcerisas@vliz.be>
