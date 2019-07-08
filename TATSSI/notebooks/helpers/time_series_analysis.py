
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

# Widgets
import ipywidgets as widgets
from ipywidgets import Layout
from ipywidgets import Select, SelectMultiple, IntProgress
from ipywidgets import Dropdown, Button, VBox, HBox
from ipywidgets import interact, interactive, fixed, interact_manual

from beakerx import TableDisplay

from IPython.display import clear_output
from IPython.display import display

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

import json
import gdal, ogr
from osgeo import gdal_array
from osgeo import osr
import pandas as pd
import xarray as xr
from rasterio import logging as rio_logging
from datetime import datetime

import matplotlib
matplotlib.use('nbAgg')

import matplotlib.pyplot as plt

class TimeSeriesAnalysis():
    """
    Class to plot a single time step and per-pixel time series
    """
    def __init__(self, qa_analytics):
        """
        :param ts: TATSSI qa_analytics object
        """
        # Clear cell
        clear_output()

        # Time series object
        self.ts = qa_analytics.ts
        # Source dir
        self.source_dir = qa_analytics.source_dir
        # Product
        self.product = qa_analytics.product
        self.version = qa_analytics.version

        # Mask
        self.mask = qa_analytics.mask

        # Data variables
        # set in __fill_data_variables
        self.data_vars = None

        # Interpolation methods
        # set in __fill_interpolation_method
        self.interpolation_methods = None

        # Display controls
        self.__display_controls()

        # Create plot objects
        self.__create_plot_objects()
        # Create plot
        self.__plot()

        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

    def __save_to_file(self, data, data_var, method):
        """
        Saves to file am interpolated time series for a specific
        data variable using a selected interpolation method
        :param data: NumPy array with the interpolated time series
        :param data_var: String with the data variable name
        :param method: String with the interpolation method name
        """
        # Get temp dataset extract the metadata
        tmp_ds = getattr(self.ts.data, data_var)
        # GeoTransform
        gt = tmp_ds.attrs['transform']
        # Coordinate Reference System (CRS) in a PROJ4 string to a
        # Spatial Reference System Well known Text (WKT)
        crs = tmp_ds.attrs['crs']
        srs = osr.SpatialReference()
        srs.ImportFromProj4(crs)
        proj = srs.ExportToWkt()

        fname = f"{self.product}.{self.version}.{data_var}.{method}.tif"
        output_dir = os.path.join(self.source_dir, data_var[1::],
                                  'interpolated')

        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)

        fname = os.path.join(output_dir, fname)

        # Get GDAL datatype from NumPy datatype
        dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)

        # Dimensions
        layers, rows, cols = data.shape

        # Create destination dataset
        dst_ds = get_dst_dataset(dst_img=fname, cols=cols, rows=rows,
                layers=layers, dtype=dtype, proj=proj, gt=gt)
        
        for layer in range(layers):
            dst_band = dst_ds.GetRasterBand(layer + 1)

            # Fill value
            dst_band.SetMetadataItem('_FillValue', str(tmp_ds.nodatavals[layer]))
            # Date
            dst_band.SetMetadataItem('RANGEBEGINNINGDATE',
                                     tmp_ds.time.data[layer].astype(str))

            # Data
            dst_band.WriteArray(data[layer])

        dst_ds = None

    def interpolate(self):
        """
        Interpolates the data of a time series object using
        the method or methods provided
        :param method: list of interpolation methods
        """
        if self.mask is None:
            pass

        # Get temp dataset to perform the interpolation
        data_var = self.data_vars.value
        tmp_ds = getattr(self.ts.data, data_var).copy(deep=True)

        _items = len(self.interpolation_methods.value) 

        progress_bar = IntProgress(
                value=0,
                min=0,
                max=_items,
                step=1,
                description='',
                bar_style='', # 'success', 'info', 'warning', 'danger' or ''
                orientation='horizontal',
                style = {'description_width': 'initial'},
                layout={'width': '75%'}
        )
        display(progress_bar)

        # For every interpol method selected by the user
        _item = 0

        progress_bar.value = _item

        # Get fill value and idx
        fill_value = tmp_ds.attrs['nodatavals'][0]
        idx_no_data = np.where(tmp_ds.data == fill_value)

        # Store original data type
        dtype = tmp_ds.data.dtype

        # Apply mask
        tmp_ds *= self.mask
        tmp_ds = tmp_ds.where(tmp_ds != 0)

        # Where there were fill values, set the value again to 
        # fill value to avoid not having data to interpolate
        tmp_ds.data[idx_no_data] = fill_value

        # Where are less than 1/4 of observations, use fill value
        min_n_obs = int(tmp_ds.shape[0] * 0.25)
        idx_lt_two_obs = np.where(self.mask.sum(axis=0) < min_n_obs)
        tmp_ds.data[:, idx_lt_two_obs[0],
                    idx_lt_two_obs[1]] = fill_value

        for method in self.interpolation_methods.value:
            progress_bar.value = _item
            progress_bar.description = (f"Interpolation of {data_var}"
                                        f" using {method}")

            tmp_interpol_ds = tmp_ds.interpolate_na(dim='time',
                                                    method=method)

            # Set data type to match the original (non-interpolated)
            tmp_interpol_ds.data = tmp_interpol_ds.data.astype(dtype)

            # Save to file
            self.__save_to_file(tmp_interpol_ds.data, data_var,
                                method)

            _item += 1

        # Remove progress bar
        progress_bar.close()
        del progress_bar

    def __create_plot_objects(self):
        """
        Create plot objects
        """
        self.fig = plt.figure(figsize=(9.0, 9.0))

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
        self.__fill_interpolation_method()

        left_box = VBox([self.data_vars])
        right_box = VBox([self.interpolation_methods])
        _HBox = HBox([left_box, right_box],
                      layout={'height': '200px',
                              'width' : '99%'}
        )
        display(_HBox)

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

    def __fill_interpolation_method(self):
        """
        Fill interpolation methods
        """
        interpolation_methods = ['linear', 'nearest', 'zero', 'slinear',
                                 'quadratic', 'cubic', 'barycentric',
                                 'krog', 'pchip', 'spline', 'akima']

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
        self.left_imshow = self.left_ds[0].plot.imshow(cmap='Greys_r', ax=self.left_p,
                             add_colorbar=False)

        # Turn off axis
        self.left_p.axis('off')
        self.left_p.set_aspect('equal')

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
        self.right_imshow = self.right_ds[0].plot.imshow(cmap='Greys_r', ax=self.right_p,
                             add_colorbar=False)

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
        # Clear subplot
        self.ts_p.clear()

        left_plot_sd = self.left_ds.sel(longitude=event.xdata,
                                        latitude=event.ydata,
                                        method='nearest')
        # Masked data
        right_plot_sd = self.right_ds.sel(longitude=event.xdata,
                                          latitude=event.ydata,
                                          method='nearest')

        # Plots
        left_plot_sd.plot(ax=self.ts_p, color='black',
                linestyle = '-', linewidth=2, label='Original data')

        # Interpolate data
        right_plot_sd_masked = right_plot_sd.where(right_plot_sd != 0)
        right_plot_sd_masked.plot(ax = self.ts_p, color='blue',
                marker='o', linestyle='None', label='Masked by user QA selection')

        # For every interpol method selected by the user
        for method in self.interpolation_methods.value:
            tmp_ds = right_plot_sd_masked.interpolate_na(dim='time',
                                              method=method)
            tmp_ds.plot(ax = self.ts_p, label=method,
                        linewidth=1)

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

