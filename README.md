# Tools for Analysing Time Series of Satellite Imagery (TATSSI)

<p align="center">
  <img src="https://raw.githubusercontent.com/GerardoLopez/TATSSI/master/TATSSI/UI/static/TATSSI.svg" alt="TATSSI logo" width="40%">
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4081163.svg)](https://doi.org/10.5281/zenodo.4081163)

## Install using Anaconda

You can install TATSSI on your favourite Linux distro or if you want to run it on Windows [here](https://github.com/GerardoLopez/TATSSI/wiki/Run-TATSSI-on-Windows-10-using-the-Windows-Subsystem-for-Linux-(WSL)) you can follow the instructions to do it.

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

## Downloading products from LP DAAC with TATSSI

Downloading products from the LP DAAC requires a NASA [EarthData](https://urs.earthdata.nasa.gov/) login. Please, first register as an EarthData user to get login credentials.

* If gedit is not installed in your system:
  * ```sudo apt install gedit```

* Update config.json file with login credentials:
  * ```cd TATSSI/TATSSI/download```
  * ```gedit config.json```
  * Replace USERNAME and PASSWORD with login credentials, save and close 

## Description

TATSSI is a set of software tools to analise Earth Observation (EO) data. It allows you to:

* Download data from the Land Processes Distributed Active Archive Center ([LP DAAC](https://lpdaac.usgs.gov/))
* Transform to/from diverse EO raster data formats using [GDAL](https://gdal.org/)
* Decode the QA-SDS associated to diverse [MODIS](https://lpdaac.usgs.gov/product_search/?collections=Combined+MODIS&collections=Terra+MODIS&collections=Aqua+MODIS&view=list) & [VIIRS](https://lpdaac.usgs.gov/product_search/?query=VIIRS&collections=S-NPP+VIIRS) data.
* Create time series of the aforementioned products masking by the user-defined QA parameter selection
* Perform basic gap-filling using the interpolation methods used in [SciPy](https://docs.scipy.org/doc/scipy/reference/interpolate.html).
* Smooth time series using robust spline smoothing following [Garcia. 2010](https://doi.org/10.1016/j.csda.2009.09.020)
* Analyse time series using different tools such as decomposition, climatologies, trends, change point detection, etc.

There are some [Jupyter Notebooks](https://jupyter.org/) associated to each module, [here](https://github.com/GerardoLopez/TATSSI/wiki/Use-TATSSI-Jupyter-Notebooks) you can find a description of each one.

## First workshop presentations (In Spanish)
* [Introducción al manejo de calidad de datos](presentaciones/IntroduccionManejoCalidadDeDatos.pptx)
* [Introducción a TATSSI](presentaciones/IntroduccionTATSSI.pptx)
* [Aplicaciones del análisis de series de tiempo](presentaciones/AplicasionesSeriesTiempo.pdf)
* [Análisis de algunos métodos de interpolación](presentaciones/AnalisisMetodosInterpolacion.pdf)

## Second workshop videos (In Spanish)
* [First day](https://www.youtube.com/watch?v=zNnw0WbnIoo&ab_channel=BiodiversidadMexicana) showing the ```Downloaders```, ```Time Series Generation```, ```QA Analytics```, ```Interpolation``` and ```Smoothing``` TATSSI modules.
* [Second day](https://www.youtube.com/watch?v=2S6J-8b7z4k&t=4739s&ab_channel=BiodiversidadMexicana) showing the ```Time Series Analysis``` TATSSI module.

## Some plots and presentations...
* A quick glimpse of a simple [plot](https://gerardolopez.github.io/TATSSI/TATSSI/scratch/plotty/VI_QA.html) for EVI and associated QAs 
* Presentations at the 2020 Joint Statistical Meetings: [Gerardo Lopez Saldana](https://docs.google.com/presentation/d/1H50s65jyT2G8JmNj8m0BRneFYpD1Ze7UO7lhbnmYaNg/edit?usp=sharing); [Inder Tecuapetla](https://irt466.wixsite.com/inder)

### Funding
TATSSI is funded by "Convocatoria de Proyectos de Desarrollo Científico para Atender Problemas Nacionales 2016" Project No. 2760; P.I.: Inder Tecuapetla. Collaborators: Gerardo Lopez Saldana, Rainer Ressl and Isabel Cruz.
