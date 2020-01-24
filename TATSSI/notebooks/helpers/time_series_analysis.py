
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
from TATSSI.time_series.mk_test import mk_test

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
from dask.diagnostics import ProgressBar

import matplotlib
import matplotlib.dates as mdates
#matplotlib.use('nbAgg')
import seaborn as sbn

import matplotlib.pyplot as plt

# Experimental
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from rpy2.robjects import numpy2ri

class TimeSeriesAnalysis():
    """
    Class to plot a single time step and per-pixel time series
    """
    debug_view = widgets.Output(layout={'border': '1px solid black'})

    def __init__(self, fname, cmap='YlGn'):
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

        # Default colormap
        self.cmap = cmap
        # Create plot objects
        self.__create_plot_objects()
        # Create plot
        self.__plot()

        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

        self.cpt = importr('changepoint')

    def __create_plot_objects(self):
        """
        Create plot objects
        """
        years_fmt = mdates.DateFormatter('%Y')

        self.fig = plt.figure(figsize=(11.0, 6.0))

        # subplot2grid((rows,cols), (row,col)
        # Left plot
        self.left_p = plt.subplot2grid((4, 4), (0, 0), rowspan=2)
        self.right_p = plt.subplot2grid((4, 4), (0, 1), rowspan=2,
                sharex=self.left_p, sharey=self.left_p)
        self.climatology = plt.subplot2grid((4, 4), (2, 0),
                rowspan=2, colspan=2)

        # Time series plot
        self.observed = plt.subplot2grid((4, 4), (0, 2), colspan=2)
        self.observed.xaxis.set_major_formatter(years_fmt)

        self.trend = plt.subplot2grid((4, 4), (1, 2), colspan=2,
                sharex=self.observed)
        self.seasonal = plt.subplot2grid((4, 4), (2, 2), colspan=2,
                sharex=self.observed)
        self.resid = plt.subplot2grid((4, 4), (3, 2), colspan=2,
                sharex=self.observed)

        #self.fig.subplots_adjust(hspace=0)

    def __display_controls(self):
        """
        Display widgets in an horizontal box
        """
        self.__fill_data_variables()
        self.__fill_year()
        self.__fill_model()

        left_box = VBox([self.data_vars, self.years])
        right_box = VBox([self.model])
        #_HBox = HBox([left_box, center_box, right_box],
        #              layout={'height': '200px',
        #                      'width' : '99%'}
        #)
        _HBox = HBox([left_box, right_box],
                     layout={'width' : '99%'})
        display(_HBox)

    def __fill_year(self):
        """
        Fill years list based on years in the dataset
        """
        # Get unique years from data
        times = getattr(self.ts.data, self.data_vars.value).time
        times = np.unique(times.dt.year.data).tolist()

        self.years = widgets.Dropdown(
                options=times,
                value=times[1],
                description='Years in time series:',
                disabled=False,
                style = {'description_width': 'initial'},
                layout={'width': '300px'}
        )

        self.years.observe(self.on_years_change)

        time_slice = slice(f"{self.years.value}-01-01",
                               f"{self.years.value}-12-31",)
        self.single_year_ds = getattr(self.ts.data,
                self.data_vars.value).sel(time=time_slice)

    def __fill_model(self):
        """
        Fill time series decompostion model
        """
        _models = ['additive', 'multiplicative']
        self.model = widgets.Dropdown(
                options=_models,
                value=_models[0],
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

    def on_years_change(self, change):
        """
        Handles a change in the years to display
        """
        if change['type'] == 'change' and change['name'] == 'value':
            time_slice = slice(f"{self.years.value}-01-01",
                               f"{self.years.value}-12-31",)
            self.single_year_ds = getattr(self.ts.data,
                    self.data_vars.value).sel(time=time_slice)

            # Update images with current year
            self.__update_imshow()

            # Redraw plot
            plt.draw()

    def __update_imshow(self):
        """
        Update images shown as imshow plots
        """
        # Plot layers at 1/3 and 2/3 of time series
        layers = self.single_year_ds.shape[0]
        first_layer = int(layers * 0.3)
        second_layer = int(layers * 0.6)

        self.left_imshow = self.single_year_ds[first_layer].plot.imshow(
                cmap=self.cmap, ax=self.left_p, add_colorbar=False)

        self.right_imshow = self.single_year_ds[second_layer].plot.imshow(
                cmap=self.cmap, ax=self.right_p, add_colorbar=False)

        self.left_p.set_aspect('equal')
        self.right_p.set_aspect('equal')

    def __plot(self):
        """
        Plot a variable and time series
        :param left_ds: xarray to plot on the left panel
        :param right_ds: xarray to plot on the right panel
        """

        # Create plot
        self.left_ds = getattr(self.ts.data, self.data_vars.value)
        self.right_ds = getattr(self.ts.data, self.data_vars.value)

        # Show images
        self.__update_imshow()

        # Turn off axis
        self.left_p.axis('off')
        self.right_p.axis('off')
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
                model=self.model.value,
                freq=self.single_year_ds.shape[0],
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

    @debug_view.capture(clear_output=True)
    def on_click(self, event):
        """
        Event handler
        """
        # Event does not apply for time series plot
        # Check if the click was in a
        if event.inaxes not in [self.left_p, self.right_p]:
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
            del self.right_p.lines[0]

        # Draw a point as a reference
        self.left_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=7, alpha=0.7)
        self.right_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=7, alpha=0.7)

        # Non-masked data
        left_plot_sd = self.left_ds.sel(longitude=event.xdata,
                                        latitude=event.ydata,
                                        method='nearest')
        # Sinlge year dataset
        single_year_ds = self.single_year_ds.sel(longitude=event.xdata,
                                                 latitude=event.ydata,
                                                 method='nearest')

        if left_plot_sd.chunks is not None:
            left_plot_sd = left_plot_sd.compute()
            single_year_ds = single_year_ds.compute()


        # Seasonal decompose
        ts_df = left_plot_sd.to_dataframe()
        self.seasonal_decompose = seasonal_decompose(
                ts_df[self.data_vars.value],
                model=self.model.value,
                freq=self.single_year_ds.shape[0],
                extrapolate_trend='freq')

        # Plot seasonal decompose
        self.observed.plot(self.seasonal_decompose.observed.index,
                self.seasonal_decompose.observed.values,
                label='Observed')

        # MK test
        _mk_test = mk_test(self.seasonal_decompose.trend)
        self.trend.plot(self.seasonal_decompose.trend.index,
                self.seasonal_decompose.trend.values,
                label=f'Trend {_mk_test}')

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
                ts_df[self.data_vars.value], ax=self.climatology)
        # Plot year to analyse
        single_year_df = single_year_ds.to_dataframe()
        sbn.stripplot(single_year_df.index.dayofyear,
                single_year_df[self.data_vars.value],
                color='red', marker='o', size=7, alpha=0.7,
                ax=self.climatology)

        self.climatology.tick_params(axis='x', rotation=70)

        # Change point
        r_vector = FloatVector(self.seasonal_decompose.trend.values)
        #changepoint_r = self.cpt.cpt_mean(r_vector)
        #changepoints_r = self.cpt.cpt_var(r_vector, method='PELT',
        #        penalty='Manual', pen_value='2*log(n)')
        changepoints_r = self.cpt.cpt_meanvar(r_vector,
                test_stat='Normal', method='BinSeg', penalty="SIC")
        changepoints = numpy2ri.rpy2py(self.cpt.cpts(changepoints_r))

        if changepoints.shape[0] > 0:
            # Plot vertical line where the changepoint was found
            for i, i_cpt in enumerate(changepoints):
                i_cpt = int(i_cpt) + 1
                cpt_index = self.seasonal_decompose.trend.index[i_cpt]
                if i == 0 :
                    self.trend.axvline(cpt_index, color='black',
                            lw='1.0', label='Change point')
                else:
                    self.trend.axvline(cpt_index, color='black', lw='1.0')

        # Legend
        self.observed.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.observed.set_title('Time series decomposition')
        self.trend.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.seasonal.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.resid.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        #self.climatology.legend([self.years.value], loc='best',
        self.climatology.legend(loc='best',
                fontsize='small', fancybox=True, framealpha=0.5)
        self.climatology.set_title('Climatology')

        # Grid
        self.observed.grid(axis='both', alpha=.3)
        self.trend.grid(axis='both', alpha=.3)
        self.seasonal.grid(axis='both', alpha=.3)
        self.resid.grid(axis='both', alpha=.3)
        self.climatology.grid(axis='both', alpha=.3)

        # Redraw plot
        plt.draw()

    def get_climatology(self, tile_size=256, n_workers=1,
                    threads_per_worker=8, memory_limit='14GB'):
        """
        Derives a climatology dataset
        """
        self.ts.climatology()

        with ProgressBar():
            self.ts.climatology_mean = self.ts.climatology_mean.compute()
            self.ts.climatology_std = self.ts.climatology_std.compute()

    @staticmethod
    def __enhance(data):

        # Histogram
        _histogram = np.histogram(data)[1]

        # Change ylimits
        max_val = _histogram[-3]
        min_val = _histogram[2]

        return min_val, max_val

