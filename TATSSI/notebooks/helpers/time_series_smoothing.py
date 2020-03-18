
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
import statsmodels.tsa.api as tsa

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
    debug_view = widgets.Output(layout={'border': '1px solid black'})

    def __init__(self, fname, band=1, isNotebook=True):
        """
        :param ts: TATSSI qa_analytics object
        """
        # Clear cell
        clear_output()

        # Time series object
        self.ts = Analysis(fname=fname)

        self.isNotebook = isNotebook
        if self.isNotebook is True:
            # Smoothing methods
            # set in __fill_smoothing_method
            self.smoothing_methods = None

            # Display controls
            self.__display_controls()

        # Create plot objects
        self.__create_plot_objects()
        # Create plot
        self.__plot(band)

        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

    def __create_plot_objects(self):
        """
        Create plot objects
        """
        self.fig = plt.figure(figsize=(10.0, 3.0))

        # Image plot
        self.img_p = plt.subplot2grid((1, 4), (0, 0), colspan=1)
        # Time series plot
        self.ts_p = plt.subplot2grid((1, 4), (0, 1), colspan=3)

    def __display_controls(self):
        """
        Display widgets in an horizontal box
        """
        self.__fill_data_variables()
        self.__fill_smoothing_method()
        self.__fill_smooth_factor()

        left_box = VBox([self.data_vars])
        center_box = VBox([self.smoothing_methods])
        right_box = VBox([self.smooth_factor])
        #_HBox = HBox([left_box, center_box, right_box],
        _HBox = HBox([left_box, center_box, right_box],
                      layout={'height': '80px',
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
                layout={'width': '150px'}
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

    def __fill_smoothing_method(self):
        """
        Fill smooth methods
        """
        smoothing_methods = ['smoothn',
                             'ExponentialSmoothing',
                             'SimpleExpSmoothing',
                             'Holt']

        self.smoothing_methods = SelectMultiple(
                options=tuple(smoothing_methods),
                value=tuple([smoothing_methods[0]]),
                rows=len(smoothing_methods),
                description='Smoothing methods',
                disabled=False,
                style = {'description_width': 'initial'},
                layout={'width': '330px'},
        )

    def __plot(self, band, is_qa=False):
        """
        Plot a variable and time series
        """

        self.img_ds = getattr(self.ts.data, self.data_vars.value)
        # Create plot
        self.img_imshow = self.img_ds[band].plot.imshow(cmap='Greys_r',
                ax=self.img_p, add_colorbar=False)

        # Turn off axis
        self.img_p.axis('off')
        self.img_p.set_aspect('equal')
        self.fig.canvas.draw_idle()

        # Connect the canvas with the event
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.on_click)

        # Plot the centroid
        _layers, _rows, _cols = self.img_ds.shape
        # Get y-axis max and min
        #y_min, y_max = self.ds.data.min(), self.ds.data.max()

        plot_sd = self.img_ds[:, int(_cols / 2), int(_rows / 2)]
        plot_sd.plot(ax = self.ts_p, color='black',
                linestyle = '-', linewidth=1, label='Original data')


        plt.margins(tight=True)
        plt.tight_layout()

        # Legend
        self.ts_p.legend(loc='best', fontsize='small',
                         fancybox=True, framealpha=0.5)

        # Grid
        self.ts_p.grid(axis='both', alpha=.3)

        plt.show()

    @debug_view.capture(clear_output=True)
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
        if len(self.img_p.lines) > 0:
            del self.img_p.lines[0]

        # Draw a point as a reference
        self.img_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=7, alpha=0.7)

        # Interpolated data to smooth
        img_plot_sd = self.img_ds.sel(longitude=event.xdata,
                                        latitude=event.ydata,
                                        method='nearest')
        if img_plot_sd.chunks is not None:
            img_plot_sd = img_plot_sd.compute()

        # Plots
        img_plot_sd.plot(ax=self.ts_p, color='black',
                linestyle = '-', linewidth=1, label='Original data')

        # For every smoothing method selected by the user
        for method in self.smoothing_methods.value:
            y = img_plot_sd.data
            s = float(self.smooth_factor.value)

            if method is 'smoothn':
                # Smoothing
                fittedvalues = smoothn(y, isrobust=True,
                        s=s, TolZ=1e-6, axis=0)[0]

            else:
                _method = getattr(tsa, method)
                # Smoothing
                #fit = _method(y).fit(smoothing_level=s, optimized=False)
                fit = _method(y.astype(float)).fit(smoothing_level=s)
                # Re-cast to original data type
                fittedvalues = np.zeros_like(fit.fittedvalues)
                fittedvalues[0:-1] = fit.fittedvalues[1::]
                fittedvalues[-1] = y[-1]

            # Plot
            tmp_ds = img_plot_sd.copy(deep=True,
                        data=fittedvalues)
            tmp_ds.plot(ax = self.ts_p, label=method, linewidth=2)

        # Change ylimits
        max_val = img_plot_sd.data.max()
        min_val = img_plot_sd.data.min()

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

