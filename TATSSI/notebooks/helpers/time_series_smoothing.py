
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(__file__)
src_dir = Path(current_dir).parents[2]
sys.path.append(str(src_dir.absolute()))

from TATSSI.time_series.smoothn import smoothn
from TATSSI.input_output.translate import Translate
from TATSSI.input_output.utils import *
from TATSSI.time_series.analysis import Analysis

# Smoothing methods
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.api import Holt

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

import matplotlib
matplotlib.use('nbAgg')

import matplotlib.pyplot as plt

class TimeSeriesSmoothing():
    """
    Class to plot a single time step and per-pixel time series
    """
    def __init__(self, fname):
        """
        :param ts: TATSSI qa_analytics object
        """
        # Clear cell
        clear_output()

        # Time series object
        self.ts = Analysis(fname=fname)

        # Smoothing methods
        # set in __fill_smoothing_method
        self.smoothing_methods = None

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
        self.fig = plt.figure(figsize=(8.0, 7.0))

        # Left plot
        self.left_p = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        # Right plot
        self.right_p = plt.subplot2grid((2, 2), (0, 1), colspan=1,
                                   sharex=self.left_p, sharey=self.left_p)
        # Time series plot
        self.ts_p = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    def __display_controls(self):
        """
        Display widgets in an horizontal box
        """
        self.__fill_data_variables()
        self.__fill_smooth_method()
        self.__fill_smooth_factor()

        left_box = VBox([self.data_vars])
        right_box = VBox([self.smooth_methods, self.smooth_factor])
        #_HBox = HBox([left_box, center_box, right_box],
        _HBox = HBox([left_box, right_box],
                      layout={'height': '180px',
                              'width' : '99%'}
        )
        display(_HBox)

    def __fill_smooth_factor(self):
        """
        Fill smooth factor bounded float text
        """
        self.smooth_factor = widgets.BoundedFloatText(
                value=0.75,
                min=0.1,
                max=10.0,
                step=0.05,
                description='Smooth factor:',
                disabled=False,
                style = {'description_width': 'initial'},
                layout={'width': '200px'}
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

    def __fill_smooth_method(self):
        """
        Fill smooth methods
        """
        interpolation_methods = ['smoothn', 'ExponentialSmoothing',
                                 'SimpleExpSmoothing', 'Holt']

        self.interpolation_methods = SelectMultiple(
                options=tuple(interpolation_methods),
                value=tuple([interpolation_methods[0]]),
                rows=len(interpolation_methods),
                description='Interpolation methods',
                disabled=False,
                style = {'description_width': 'initial'},
                layout={'width': '220px'},
        )

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
        #vmin, vmax = self.__enhance(self.left_ds[0].data)
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
        # Get y-axis max and min
        #y_min, y_max = self.ds.data.min(), self.ds.data.max()

        plot_sd = self.left_ds[:, int(_cols / 2), int(_rows / 2)]
        plot_sd.plot(ax = self.ts_p, color='black',
                linestyle = '--', linewidth=1, label='Original data')


        # Right panel
        if self.mask is None:
            self.right_ds = self.left_ds.copy(deep=True)
        else:
            self.right_ds = self.left_ds * self.mask

        # Create plot
        #self.right_ds[0].plot(cmap='Greys_r', ax=self.right_p,
        #                      add_colorbar=False)
        self.right_imshow = self.right_ds[0].plot.imshow(cmap='Greys_r',
                ax=self.right_p, add_colorbar=False)

        # Turn off axis
        self.right_p.axis('off')
        self.right_p.set_aspect('equal')

        plt.margins(tight=True)
        plt.tight_layout()

        # Legend
        self.ts_p.legend(loc='best', fontsize='small',
                         fancybox=True, framealpha=0.5)

        plt.show()

    def on_click(self, event):
        """
        Event handler
        """
        # Event does not apply for time series plot
        # Check if the click was in a
        if event.inaxes in [self.ts_p]:
            return

        # Clear subplot
        self.ts_p.clear()

        # Delete last reference point
        if len(self.left_p.lines) > 0:
            del self.left_p.lines[0]
            del self.right_p.lines[0]

        # Draw a point as a reference
        self.left_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=3)
        self.right_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=3)

        # Non-masked data
        left_plot_sd = self.left_ds.sel(longitude=event.xdata,
                                        latitude=event.ydata,
                                        method='nearest')
        if left_plot_sd.chunks is not None:
            left_plot_sd = left_plot_sd.compute()

        # Masked data
        right_plot_sd = self.right_ds.sel(longitude=event.xdata,
                                          latitude=event.ydata,
                                          method='nearest')
        if right_plot_sd.chunks is not None:
            right_plot_sd = right_plot_sd.compute()

        # Plots
        left_plot_sd.plot(ax=self.ts_p, color='black',
                linestyle = '-', linewidth=1, label='Original data')

        # Interpolate data
        right_plot_sd_masked = right_plot_sd.where(right_plot_sd != 0)
        right_plot_sd_masked.plot(ax = self.ts_p, color='blue',
                marker='o', linestyle='None', alpha=0.7, markersize=4,
                label='Masked by user QA selection')

        # For every interpol method selected by the user
        for method in self.interpolation_methods.value:
            if method is 'smoothn':
                # Linear interpolation
                y = right_plot_sd_masked.interpolate_na(dim='time').data
                # Weigth obs
                idx = np.nonzero(right_plot_sd.data)
                w = right_plot_sd.copy(deep=True).data
                w[idx] *= 2
                # Smoothing
                s = float(self.smooth_factor.value)
                smoothed_array = smoothn(y, W=w, isrobust=True,
                        s=s, TolZ=1e-6, axis=0)

                tmp_ds = right_plot_sd_masked.copy(deep=True,
                        data=smoothed_array[0])
            else:
                tmp_ds = right_plot_sd_masked.interpolate_na(dim='time',
                                              method=method)

            # Plot
            tmp_ds.plot(ax = self.ts_p, label=method, linewidth=2)

        # Change ylimits
        max_val = left_plot_sd.data.max()
        min_val = left_plot_sd.data.min()

        data_range = max_val - min_val
        max_val = max_val + (data_range * 0.2)
        min_val = min_val - (data_range * 0.2)
        self.ts_p.set_ylim([min_val, max_val])

        # Legend
        self.ts_p.legend(loc='best', fontsize='small',
                         fancybox=True, framealpha=0.5)

        # Grid
        self.ts_p.grid(axis='both', alpha=.3)

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

