
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(__file__)
src_dir = Path(current_dir).parents[2]
sys.path.append(str(src_dir.absolute()))

from TATSSI.input_output.translate import Translate
from .utils import *
from TATSSI.time_series.analysis import Analysis

from statsmodels.tsa.seasonal import seasonal_decompose

# Widgets
import ipywidgets as widgets
from ipywidgets import Layout
from ipywidgets import Select, SelectMultiple, IntProgress
from ipywidgets import Dropdown, Button, VBox, HBox, BoundedFloatText
from ipywidgets import interact, interactive, fixed, interact_manual

from beakerx import TableDisplay

from IPython.display import clear_output
from IPython.display import display

import json
import gdal, ogr
from osgeo import gdal_array
from osgeo import osr
import pandas as pd
import xarray as xr
import numpy as np
from rasterio import logging as rio_logging
from datetime import datetime

from dask.distributed import Client

import matplotlib
import matplotlib.dates as mdates
#matplotlib.use('nbAgg')
import seaborn as sbn

import matplotlib.pyplot as plt

class TimeSeriesAnalysis():
    """
    Class to plot a single time step and per-pixel time series
    """
    def __init__(self, fname):
        """
        :param ts: TATSSI file time series
        """
        # Clear cell
        clear_output()

        # Time series object
        self.ts = Analysis(fname=fname)

        # Data variables
        # set in __fill_data_variables
        self.data_vars = None

        # Display controls
        self.__display_controls()

        # Create plot objects
        self.__create_plot_objects()
        # Create plot
        self.__plot()

        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

    def __create_plot_objects(self):
        """
        Create plot objects
        """
        years_fmt = mdates.DateFormatter('%Y')

        self.fig = plt.figure(figsize=(11.0, 6.0))

        # subplot2grid((rows,cols), (row,col)
        # Left plot
        self.left_p = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
        self.climatology = plt.subplot2grid((4, 2), (3, 0), rowspan=1)

        # Time series plot
        self.observed = plt.subplot2grid((4, 2), (0, 1), colspan=1)
        self.observed.xaxis.set_major_formatter(years_fmt)

        self.trend = plt.subplot2grid((4, 2), (1, 1), colspan=1,
                sharex=self.observed)
        self.seasonal = plt.subplot2grid((4, 2), (2, 1), colspan=1,
                sharex=self.observed)
        self.resid = plt.subplot2grid((4, 2), (3, 1), colspan=1,
                sharex=self.observed)

    def __display_controls(self):
        """
        Display widgets in an horizontal box
        """
        self.__fill_data_variables()
        #self.__fill_interpolation_method()
        self.__fill_model()

        left_box = VBox([self.data_vars])
        #center_box = VBox([self.interpolation_methods])
        right_box = VBox([self.model])
        #_HBox = HBox([left_box, center_box, right_box],
        #              layout={'height': '200px',
        #                      'width' : '99%'}
        #)
        _HBox = HBox([left_box, right_box],
                     layout={'width' : '99%'})
        display(_HBox)

    def __fill_model(self):
        """
        Fill time series decompostion model
        """
        self.model = widgets.Dropdown(
                options=['multiplicative', 'additive'],
                value='multiplicative',
                description='Decompostion model:',
                disabled=False,
                style = {'description_width': 'initial'},
                layout={'width': '300px'}
        )

    def __fill_data_variables(self):
        """
        Fill the data variables dropdown list
        """
        data_vars = []
        for data_var in self.ts.data.data_vars:
            data_vars.append(data_var)

        self.data_vars = Dropdown(
            options=data_vars,
            value=data_vars[0],
            description='Data variables:',
            disabled=False,
            style = {'description_width': 'initial'},
            layout={'width': '400px'},
        )

        self.data_vars.observe(self.on_data_vars_change)

    def on_data_vars_change(self, change):
        """
        Handles a change in the data variable to display
        """
        if change['type'] == 'change' and change['name'] == 'value':
            self.left_ds = getattr(self.ts.data, change['new'])
            if self.mask is None:
                self.right_ds = self.left_ds.copy(deep=True)
            else:
                self.right_ds = self.left_ds * self.mask

            self.left_imshow.set_data(self.left_ds.data[0])
            self.right_imshow.set_data(self.right_ds.data[0])

    def __plot(self, is_qa=False):
        """
        Plot a variable and time series
        :param left_ds: xarray to plot on the left panel
        :param right_ds: xarray to plot on the right panel
        """

        self.left_ds = getattr(self.ts.data, self.data_vars.value)
        # Create plot
        #self.left_ds[0].plot(cmap='Greys_r', ax=self.left_p,
        #                     add_colorbar=False)
        #vmin, vmax = self.__enhance(np.copy(self.left_ds[0].data))
        self.left_imshow = self.left_ds[0].plot.imshow(cmap='Greys_r',
                ax=self.left_p, add_colorbar=False)

        # Turn off axis
        self.left_p.axis('off')
        self.left_p.set_aspect('equal')
        self.fig.canvas.draw_idle()

        # Connect the canvas with the event
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.on_click)

        # Plot the centroid
        _layers, _rows, _cols = self.left_ds.shape

        # Seasonal decompose
        left_plot_sd = self.left_ds[:, int(_cols / 2), int(_rows / 2)]
        ts_df = left_plot_sd.to_dataframe()
        self.seasonal_decompose = seasonal_decompose(
                ts_df[self.data_vars.value],
                model=self.model.value, freq=23,
                extrapolate_trend='freq')

        # Plot seasonal decompose
        self.seasonal_decompose.observed.plot(ax=self.observed)
        self.seasonal_decompose.trend.plot(ax=self.trend)
        self.seasonal_decompose.seasonal.plot(ax=self.seasonal)
        self.seasonal_decompose.resid.plot(ax=self.resid)

        # Climatology
        sbn.boxplot(ts_df.index.dayofyear,
                ts_df[self.data_vars.value],
                ax=self.climatology)
        self.climatology.tick_params(axis='x', rotation=70)

        #plot_sd = self.left_ds[:, int(_cols / 2), int(_rows / 2)]
        #plot_sd.plot(ax = self.ts_p, color='black',
        #        linestyle = '--', linewidth=1, label='Original data')

        plt.margins(tight=True)
        plt.tight_layout()

        # Legend
        #self.ts_p.legend(loc='best', fontsize='small',
        #                 fancybox=True, framealpha=0.5)

        plt.show()

    def on_click(self, event):
        """
        Event handler
        """
        # Event does not apply for time series plot
        # Check if the click was in a
        if event.inaxes not in [self.left_p]:
            return

        # Clear subplots
        self.observed.clear()
        self.trend.clear()
        self.seasonal.clear()
        self.resid.clear()
        self.climatology.clear()

        # Delete last reference point
        if len(self.left_p.lines) > 0:
            del self.left_p.lines[0]

        # Draw a point as a reference
        self.left_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=3)

        # Non-masked data
        left_plot_sd = self.left_ds.sel(longitude=event.xdata,
                                        latitude=event.ydata,
                                        method='nearest')
        if left_plot_sd.chunks is not None:
            left_plot_sd = left_plot_sd.compute()

        # Seasonal decompose
        ts_df = left_plot_sd.to_dataframe()
        self.seasonal_decompose = seasonal_decompose(
                ts_df[self.data_vars.value],
                model=self.model.value, freq=23,
                extrapolate_trend='freq')

        # Plot seasonal decompose
        self.observed.plot(self.seasonal_decompose.observed.index,
                self.seasonal_decompose.observed.values,
                label='Observed')
        self.trend.plot(self.seasonal_decompose.trend.index,
                self.seasonal_decompose.trend.values,
                label='Trend')

        # Set the same y limits from observed data
        self.trend.set_ylim(self.observed.get_ylim())

        self.seasonal.plot(self.seasonal_decompose.seasonal.index,
                self.seasonal_decompose.seasonal.values,
                label='Seasonality')
        self.resid.plot(self.seasonal_decompose.resid.index,
                self.seasonal_decompose.resid.values,
                label='Residuals')

        # Climatology
        sbn.boxplot(ts_df.index.dayofyear,
                #left_plot_sd, ax=self.climatology)
                ts_df[self.data_vars.value], ax=self.climatology)
        self.climatology.tick_params(axis='x', rotation=70)

        # Legend
        self.observed.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.trend.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.seasonal.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.resid.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.climatology.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)

        # Grid
        self.observed.grid(axis='both', alpha=.3)
        self.trend.grid(axis='both', alpha=.3)
        self.seasonal.grid(axis='both', alpha=.3)
        self.resid.grid(axis='both', alpha=.3)
        self.climatology.grid(axis='both', alpha=.3)

        # Redraw plot
        plt.draw()

    @staticmethod
    def __enhance(data):

        # Histogram
        _histogram = np.histogram(data)[1]

        # Change ylimits
        max_val = _histogram[-3]
        min_val = _histogram[2]

        return min_val, max_val

