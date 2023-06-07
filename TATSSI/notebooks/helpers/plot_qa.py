
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(__file__)
src_dir = Path(current_dir).parents[2]
sys.path.append(str(src_dir.absolute()))

from TATSSI.time_series.generator import Generator
from TATSSI.input_output.translate import Translate
from TATSSI.input_output.utils import *
from TATSSI.qa.EOS.catalogue import Catalogue

from TATSSI.download.modis_downloader import get_modis_data
from TATSSI.download.viirs_downloader import get_viirs_data

# Widgets
import ipywidgets as widgets
from ipywidgets import Layout
from ipywidgets import Button, HBox, VBox
from ipywidgets import interact, interactive, fixed, interact_manual

from beakerx import TableDisplay

from IPython.display import clear_output
from IPython.display import display

import json
from osgeo import gdal, ogr
import pandas as pd
import xarray as xr
from rasterio import logging as rio_logging
from datetime import datetime

import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

class PlotQA():
    """
    Class to handle QA (mainly) plotting within a Jupyter Notebook
    """
    def __init__(self):
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(9.5,11))

        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

    def _get_colorbar(self, ticks, labels):
        """
        Creates colorbar
        :param ticks: array with ticks
        :param labels: list with description for every tick
        """
        cbar = self.fig.colorbar(self._plot,
                                 orientation='horizontal',
                                 pad=0.05)

        cbar.set_ticks(ticks)
        cbar.ax.set_xticklabels(labels, rotation=20,
                                fontdict={'fontsize': 8})

        return cbar

    def plot(self, qa_fname):
        """
        Plot QA flags using raster category names
        """
        # Open dataset
        qa = xr.open_rasterio(qa_fname)

        # Get categories
        ds = gdal.Open(qa_fname)
        band = ds.GetRasterBand(1)

        # Get raster attribute table
        band.GetRasterColorTable()
        rat = band.GetDefaultRAT()

        category_names = rat.ReadAsArray(1).astype(str)

        ## band.GetRasterColorTable() # Needed to get category names
        ## category_names = np.array(band.GetRasterCategoryNames())

        # Get indices where there is a category description
        idx = np.where(category_names != '')

        # Create plot
        self._plot = qa[0].plot(levels = np.append(idx[0], idx[0][-1]),
                                shading = 'flat',
                                cmap = 'viridis_r',
                                add_colorbar = False,
                                ax = self.ax)

        cbar = self._get_colorbar(idx[0], category_names[idx].tolist())

        #self.ax.format_coord = format_coord
        self.ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()
