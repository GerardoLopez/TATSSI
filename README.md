# Tools for Analysing Time Series of Satellite Imagery (TATSSI)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4050082.svg)](https://doi.org/10.5281/zenodo.4050082)


## Install using Anaconda

* Download conda
  * ```wget https://repo.continuum.io/archive/Anaconda3-2019.07-Linux-x86_64.sh```
    * Install conda ```bash ./Anaconda3-2019.07-Linux-x86_64.sh```
      * Accept the default settings. When asked:
        ```Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]```
        Say: ```yes```
    * Close that shell and open a new one
* Clone this repo
  * ```git clone https://github.com/GerardoLopez/TATSSI```
* Install the required libraries:
  * ```cd TATSSI```
  * ```conda install --file tatssi-package-list.txt```
  * If you wanto to use the [```changepoint```](http://dx.doi.org/10.18637/jss.v058.i03) R package:
    * Install R
      * ```sudo apt update```
      * ```sudo apt-get install r-base```
    * Install the ```changepoint``` package
      * Run R with the following command: ```/usr/bin/R```
      * ```install.packages('changepoint')```
      * ```install.packages('changepoint.np')```
      * Exit R with the following command: ```quit()```

* Run TATSSI
  * If you want to use the Jupyter Notebooks:
    * Go to the ```TATSSI/notebooks``` directory and run ```jupyter notebook```
  * If you prefer to use the UI:
    * Go to the ```TATSSI/TATSSI/UI``` directory and run ```python tatssi.py```

## Description

TATSSI is a set of software tools to analise Earth Observation (EO) data. It allows you to:

* Download data from the Land Processes Distributed Active Archive Center ([LP DAAC](https://lpdaac.usgs.gov/))
* Transform to/from diverse EO raster data formats using [GDAL](https://gdal.org/)
* Decode the QA-SDS associated to diverse [MODIS](https://lpdaac.usgs.gov/product_search/?collections=Combined+MODIS&collections=Terra+MODIS&collections=Aqua+MODIS&view=list) & [VIIRS](https://lpdaac.usgs.gov/product_search/?query=VIIRS&collections=S-NPP+VIIRS) data.
* Create time series of the aforemetnioned products masking by the user-defined QA parameter selection
* Perform basic gap-filling using the interpolation methods used in [SciPy](https://docs.scipy.org/doc/scipy/reference/interpolate.html).

There are some [Jupyter Notebooks](https://jupyter.org/) associated to each module, go to ```TATSSI/notebooks``` and enjoy!

## Presentaciones del primer taller
* [Introducción al manejo de calidad de datos](presentaciones/IntroduccionManejoCalidadDeDatos.pptx)
* [Introducción a TATSSI](presentaciones/IntroduccionTATSSI.pptx)
* [Aplicaciones del análisis de series de tiempo](presentaciones/AplicasionesSeriesTiempo.pdf)
* [Análisis de algunos métodos de interpolación](presentaciones/AnalisisMetodosInterpolacion.pdf)

## Some plots and presentations...
* A quick glimpse of a simple [plot](https://gerardolopez.github.io/TATSSI/TATSSI/scratch/plotty/VI_QA.html) for EVI and associated QAs 
* Presentation at [JSM2020](https://docs.google.com/presentation/d/1H50s65jyT2G8JmNj8m0BRneFYpD1Ze7UO7lhbnmYaNg/edit?usp=sharing)

### Funding
TATSSI is funded by "Convocatoria de Proyectos de Desarrollo Científico para Atender Problemas Nacionales 2016" Project No. 2760; P.I.: Inder Tecuapetla. Collaborators: Gerardo Lopez Saldana, Rainer Ressl and Isabel Cruz.
