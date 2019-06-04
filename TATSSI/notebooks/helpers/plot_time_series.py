
import os
import sys

# TATSSI modules
HomeDir = os.path.join(os.path.expanduser('~'))
SrcDir = os.path.join(HomeDir, 'Projects', 'TATSSI')
sys.path.append(SrcDir)

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

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

import json
import gdal, ogr
import pandas as pd
import xarray as xr
from rasterio import logging as rio_logging
from datetime import datetime

import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

class PlotTimeSeries():
    """
    Class to plot a single time step and per-pixel time series
    """
    def __init__(self):
        self.fig = plt.figure(figsize=(9.0, 9.0))

        # Data plot
        self.data_p = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        # QA plot
        self.qa_p = plt.subplot2grid((2, 2), (0, 1), colspan=1,
                                   sharey=self.data_p)
        # Time series plot
        self.ts_p = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

        self.ds = None

    def plot(self, ds, qa):
        """
        Plot a variable and time series
        :param fname: Full path file name of time series to plot
        """
        # Open dataset
        self.ds = ds
        # Create plot
        self.ds[0].plot(cmap='Greys_r', ax=self.data_p,
                        add_colorbar=False)
        # Turn off axis
        self.data_p.axis('off')
        self.data_p.set_aspect('equal')

        # Connect the canvas with the event
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.on_click)

        # Plot the centroid
        _layers, _rows, _cols = self.ds.shape
        # Get y-axis max and min
        #y_min, y_max = self.ds.data.min(), self.ds.data.max()

        plot_sd = self.ds.sel(longitude = int(_cols / 2),
                              latitude = int(_rows / 2),
                              method='nearest')

        plot_sd.plot(ax = self.ts_p)
        #self.bx.set_ylim([y_min, y_max])

        plt.tight_layout()
        plt.show()

    def on_click(self, event):
        """
        Event handler
        """
        # Clear subplot
        self.ts_p.clear()

        plot_sd = self.ds.sel(longitude=event.xdata,
                              latitude=event.ydata,
                              method='nearest')

        plot_sd.plot(ax = self.ts_p)
        # Redraw plot
        plt.draw()
