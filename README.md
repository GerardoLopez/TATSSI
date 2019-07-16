# Tools for Analysing Time Series of Satellite Imagery (TATSSI)

## Install using Anaconda

* Download conda
  * ```wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh```
    * Install conda ```bash ./Anaconda3-5.1.0-Linux-x86_64.sh```
* Clone this repo
  * ```git clone https://github.com/GerardoLopez/TATSSI```
* Install the required libraries:
  * ```cd TATSSI```
  * ```conda install --file tatssi-package-list.txt```

## Description

TATSSI is a set of software tools to analise Earth Observation (EO) data. It allows you to:

* Download data from the Land Processes Distributed Active Archive Center ([LP DAAC](https://lpdaac.usgs.gov/))
* Transform to/from diverse EO raster data formats using [GDAL](https://gdal.org/)
* Decode the QA-SDS associated to diverse [MODIS](https://lpdaac.usgs.gov/product_search/?collections=Combined+MODIS&collections=Terra+MODIS&collections=Aqua+MODIS&view=list) & [VIIRS](https://lpdaac.usgs.gov/product_search/?query=VIIRS&collections=S-NPP+VIIRS) data.
* Create time series of the aforemetnioned products masking by the user-defined QA parameter selection
* Perform basic gap-filling using the interpolation methods used in [SciPy](https://docs.scipy.org/doc/scipy/reference/interpolate.html).

There are some [Jupyter Notebooks](https://jupyter.org/) associated to each module, go to ```TATSSI/notebooks``` and enjoy!

## Taller

Si eres de los participantes al taller, da click [aquí](http://35.237.20.223/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FGerardoLopez%2FTATSSI&app=notebook) e ingresa con tu usuario y contraseña.

## Some plots...
A quick glimpse of a simple [plot](https://gerardolopez.github.io/TATSSI/TATSSI/scratch/plotty/VI_QA.html) for EVI and associated QAs 
