# Tools for Analysing Time Series of Satellite Imagery (TATSSI)

## Install using Anaconda

* Download conda
  * ```wget wget https://repo.continuum.io/archive/Anaconda3-2019.07-Linux-x86_64.sh```
    * Install conda ```bash ./Anaconda3-2019.07-Linux-x86_64.sh```
      * Accept the default settings. When asked:
        ```Do you wish the installer to initialize Anaconda3
           by running conda init? [yes|no]``
        Say: ```yes```
    * Close that shell and open a new one
* Clone this repo
  * ```git clone https://github.com/GerardoLopez/TATSSI```
* Install the required libraries:
  * ```cd TATSSI```
  * ```conda install --file tatssi-package-list.txt```
* Go to the ```TATSSI/notebooks``` directory and run ```jupyter notebook```

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

## Some plots...
A quick glimpse of a simple [plot](https://gerardolopez.github.io/TATSSI/TATSSI/scratch/plotty/VI_QA.html) for EVI and associated QAs 

### Funding
TATSSI is funded by "Convocatoria de Proyectos de Desarrollo Científico para Atender Problemas Nacionales 2016" Project No. 2760; P.I.: Inder Tecuapetla. Collaborators: Gerardo Lopez Saldana, Rainer Ressl and Isabel Cruz.
