
import os
import sys

# TATSSI modules
HomeDir = os.path.join(os.path.expanduser('~'))
SrcDir = os.path.join(HomeDir, 'Projects', 'TATSSI')
sys.path.append(SrcDir)

from TATSSI.input_output.translate import Translate
from TATSSI.input_output.utils import *
from TATSSI.qa import catalogue
from TATSSI.download.modis_downloader import get_modis_data
from TATSSI.download.viirs_downloader import get_viirs_data

from IPython.display import clear_output
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual

from ipywidgets import Layout
from ipywidgets import Button, HBox, VBox

import ipywidgets as widgets

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

import json
import gdal, ogr
import pandas as pd
import xarray as xr
from rasterio import logging as rio_logging

from beakerx import TableDisplay

from datetime import datetime

import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

def open_file_dialog(dialog_type = 'open',
                   data_format = 'GeoTiff',
                   extension = 'tif'):
    """
    Creates a Open File dialog window
    :param dialog_type: Dialog type, can be open or save. Default is
                        open
    :param data_format: Data format to Open/Save. Default is GeoTiff
    :parm extension: Data format extension. Default is tif
    :return: Full path of selected file
    """
    app = QtWidgets.QApplication([dir])
    if dialog_type == 'open':
        fname = QFileDialog.getOpenFileName(None,
                    "Select a file...", '.',
                    filter="All files (*)")

    elif dialog_type == 'save':
        # Get format and extension
        fname = QFileDialog.getSaveFileName(None,
                    "Save file as...", '.',
                    filter="%s Files (*.%s)" % \
                            (data_format, extension))

    elif dialog_type == 'directory':
        dirname = QFileDialog.getExistingDirectory(None)
        return dirname

    return str(fname[0])

def read_config():
    """
    Read downloaders config file
    """
    with open('config.json') as f:
        credentials = json.load(f)

    url = credentials['url']
    username = credentials['username']
    password = credentials['password']

    return url, username, password

class PlotTimeSeries():
    """
    Class to plot a single time step and per-pixel time series
    """
    def __init__(self):
        self.fig, (self.ax, self.bx) = plt.subplots(nrows = 2, ncols = 1,
                                           figsize=(7.5, 9.0))

        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

        self.ds = None

    def plot(self, ds):
        """
        Plot a variable and time series
        :param fname: Full path file name of time series to plot
        """
        # Open dataset
        self.ds = ds

        # Create plot
        self._plot = self.ds[0].plot(cmap = 'Greys_r',
                                     ax = self.ax)

        #self.ax.format_coord = format_coord
        self.ax.set_aspect('equal')

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

        plot_sd.plot(ax = self.bx)
        #self.bx.set_ylim([y_min, y_max])

        plt.tight_layout()
        plt.show()

    def on_click(self, event):
        """
        Event handler
        """
        # Clear subplot
        self.bx.clear()

        plot_sd = self.ds.sel(longitude=event.xdata,
                              latitude=event.ydata,
                              method='nearest')

        plot_sd.plot(ax = self.bx)
        # Redraw plot
        plt.draw()

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
        band.GetRasterColorTable() # Needed to get category names
        category_names = np.array(band.GetRasterCategoryNames())
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

class Download():
    """
    Class to handle donwload operations within the notebook
    """
    def __init__(self):
        # Create TATSSI catalogue object
        self.catalogue = catalogue.Catalogue()

        # Product and dates button
        self.select_product_dates_button = widgets.Button(
                layout = Layout(width='99.6%'),
                description = 'Select tile, product and dates',
                tooltip = 'Select MODIS/VIIRS tile, product to ' + \
                          'be downloaded and required dates.')

        # Output directory button
        self.output_dir_button = widgets.Button(
                layout = Layout(width='99.6%'),
                description = 'Select output directory',
                tooltip = 'Select output directory where to ' + \
                          'store products for required dates.')

        # Tiles list
        tiles = self.get_tiles_list()
        self.tiles = widgets.Dropdown(
                options = tiles,
                value = tiles[0],
                description = 'Tiles:',
                disable = False)

        self.product = None
        self.product_table = None
        self.output = None
        self.download_button = None

        self.download()

    def download(self):
        """
        Downloads a product from the LPDAAC
        """
        # Set on click event behaviour for buttons
        self.select_product_dates_button.on_click(
                self.__on_product_dates_button_clicked)

        self.output_dir_button.on_click(
                self.__on_output_dir_button_clicked)

        # Display button
        display(self.select_product_dates_button)

    def __on_product_dates_button_clicked(self, b):
        """
        Shows table with available products and enables
        the download button.
        """
        def on_product_table_double_click(row, col, table):
            """Internal function to update product and dates"""
            value = table.values[row][-1]
            self.product.value = f"<b>{value}</b>"
            # Update dates
            start = self.__string_to_datetime(table.values[row][-5])
            self.start_date.value = start
            end = self.__string_to_datetime(table.values[row][-4])
            self.end_date.value = end

        self.__clear_cell()
        # Display widgets
        display(self.select_product_dates_button)
        display(self.tiles)

        # Display products
        self.product_table = TableDisplay(self.catalogue.products)
        self.product_table.setDoubleClickAction(
                on_product_table_double_click)
        display(self.product_table)

        # Display label with product to download
        style = {'description_width': 'initial'}
        value = self.catalogue.products.ProductAndVersion.iloc[0]
        self.product = widgets.HTML(
                value = f"<b>{value}</b>",
                placeholder = "Product and version:",
                description = "Product and version:",
                layout = Layout(width = '100%'),
                style = style)

        display(self.product)

        # Dates
        self.__display_dates()

        # Output dir button
        display(self.output_dir_button)

    def __on_output_dir_button_clicked(self, b):
        """
        Opens dialog to select output dir
        """
        output_dir = open_file_dialog('directory')
        if len(output_dir) == 0:
            if self.download_button is not None:
                self.download_button.disable = True
            # If there's no output file, do nothing...
            return None

        if self.output is None:
            # Show input file in a text file
            style = {'description_width': 'initial'}
            self.output = widgets.HTML(
                value = f"<b>{output_dir}</b>",
                placeholder = "Output directory",
                description = "Output directory",
                layout = Layout(width = '100%'),
                style = style)

            display(self.output)
        else:
            self.output.value = output_dir

        if self.download_button is None:
            # Download button
            self.download_button = widgets.Button(
                    layout = Layout(width='99.6%'),
                    description = 'Download product for selected dates',
                    tooltip = 'Download selected product for ' + \
                              ' required date range.')

            self.download_button.on_click(
                    self.__on_download_button_clicked)

            display(self.download_button)
        else:
            self.download_button.disable = False

    def __on_download_button_clicked(self, b):
        """
        Launch the donwloader for user's selection of product and dates
        """
        # Get output dir, rm <b> HTML tag
        output = self.output.value[3:-4]

        # Get platform, rm <b> HTML tag
        product = self.product.value[3:-4]
        if 'VNP' in product:
            platform = 'VIIRS'
            donwloader = get_viirs_data
        else:
            platform = self.get_modis_platform(product)
            donwloader = get_modis_data

        url, username, password = read_config()        

        # Dates needs to be datetime objects
        start_date = datetime.combine(self.start_date.value,
                datetime.min.time())
        end_date = datetime.combine(self.end_date.value,
                datetime.min.time())

        # Run the downloader
        donwloader(platform = platform,
                   product = product,
                   tiles = self.tiles.value,
                   output_dir = output,
                   start_date = start_date,
                   end_date = end_date,
                   username = username,
                   password = password)

    def __display_dates(self):
        """
        Manage dates widgets
        """
        # Dates to download
        start = self.catalogue.products.TemporalExtentStart.iloc[0]
        start= self.__string_to_datetime(start)
        style = {'description_width': 'initial'}
        self.start_date = widgets.DatePicker(
                value = start,
                description = 'Select start date',
                disabled = False,
                style = {'description_width': 'initial'})

        end = self.catalogue.products.TemporalExtentEnd.iloc[0]
        end = self.__string_to_datetime(end)
        style = {'description_width': 'initial'}
        self.end_date = widgets.DatePicker(
                value = end,
                description = 'Select end date',
                disabled = False,
                style = {'description_width': 'initial'})

        display(self.start_date)
        display(self.end_date)

    def __clear_cell(self):
        """ Clear cell """
        clear_output()

    @staticmethod
    def __string_to_datetime(string_date):
        """
        Convert a string into a datetime object
        """
        try:
            # Default format should be yyyy-mm-dd
            date = datetime.strptime(string_date, '%Y-%m-%d')
        except ValueError:
            try:
                # Some dates in "full month name" day, year
                date = datetime.strptime(string_date, '%B %d, %Y')
            except ValueError:
                date = datetime.today()

        return date

    @staticmethod
    def get_tiles_list():
        """
        Gets a list of available MODIS/VIIRS tiles
        """
        current_dir = os.path.join(os.path.dirname(__file__))
        fname = os.path.join(current_dir,
                "../../../data/kmz/modis_sin.kmz")
        d = ogr.Open(fname)

        # Empty list of tiles
        tiles = []

        layer = d.GetLayer()
        for feature in layer:
            # e.g. h:4 v:7
            feature = feature.GetField(0)
            h, v = feature.split(' ')
            h = int(h.split('h:')[1].strip())
            v = int(v.split('v:')[1].strip())
            tile = f'h{h:02}v{v:02}'

            tiles.append(tile)

        tiles.sort()
        return tiles

    @staticmethod
    def get_modis_platform(modis_product):
        """
        Get MODIS plattform: MOLT, MOLA or MOTA. This basically relates
        to the sensor used (or if a combination of AQUA & TERRA is used)
        """
        product = modis_product.split('.')[0]
        if 'MCD' in product:
            return 'MOTA'
        elif 'MOD' in product:
            return 'MOLT'
        else:
            return 'MOLA'

class ImportExport():

    def __init__(self):
        """
        Class to handle Input/Output operations within the notebook
        """
        self.input_button = widgets.Button(
            description = 'Select input file',
            tooltip = 'Select file or dataset to be imported.'
        )

        self.input = None
        self.translate_button = None
        self.output_button = None
        self.output = None
        self.format = None
        self.import_export()

    def __on_output_button_clicked(self, b):
        """
        Based on user file selection displays the output file
        """
        data_format = self.format.value.split('|')[0].strip()
        extension = self.format.value.split('|')[2].strip()

        target_img = open_file_dialog('save', data_format, extension)
        if len(target_img) == 0:
            # If there's no output file, do nothing...
            return None

        if self.output is None:
            # Show input file in a text file
            self.output = widgets.Text(
                              value = target_img,
                              placeholder = "File to be exported",
                              description = "Output file",
                              layout = Layout(width='100%'))

            display(self.output)
        else:
            self.output.value = target_img

        if self.translate_button is not None:
            # Translate button already created...
            return None

        # Create translate button
        self.translate_button = widgets.Button(
            description = 'Translate',
            tooltip = 'Translate input file into output file'
        )

        self.translate_button.on_click(self.__on_translate_button_clicked)
        display(self.translate_button)

    def __on_translate_button_clicked(self, b):
        """
        Performs the translation into an output file with selected format
        """
        # Checks...
        try:
            # Use GDAL exceptions
            gdal.UseExceptions()
            tmp_d = gdal.Open(self.input.value)
        except Exception as err:
            if err.err_level >= gdal.CE_Warning:
                print('Cannot read input dataset: %s' % self.input.value)
            # Stop using GDAL exceptions
            gdal.DontUseExceptions()
            raise RuntimeError(err.err_level, err.err_no, err.err_msg)

        if not os.path.isdir(os.path.dirname(self.output.value)):
            print("Output directory %s does not exist" % \
                  os.path.dirname(self.output.value))

        # Selected GDAL driver
        driver = self.format.value.split('|')[1].strip()
        # Translate
        Translate(self.input.value, self.output.value, driver)

    def __on_input_button_clicked(self, b):
        """
        Based on user file selection displays either the input file
        or the SubDatasets of the selected file
        """
        self.__clear_cell()
        display(self.input_button)

        source_img = open_file_dialog('open')
        if len(source_img) == 0:
            # If there's no source file, do nothing...
            return None

        if has_subdatasets(source_img):
            # Get SubDatasets
            sds = get_subdatasets(source_img)
            sds_df = pd.DataFrame(list(sds))
            sds_df.columns = ['SubDataset', 'Resolution']

            # Put SubDatasets in a DropDown list
            self.input = widgets.Dropdown(
                             options = list(sds_df['SubDataset']),
                             value = sds_df['SubDataset'][0],
                             description = "SubDataset",
                             layout=Layout(width='100%'))
        else:
            # Show input file in a text file
            self.input = widgets.Text(
                              value = source_img,
                              placeholder = "File to be imported",
                              description = "Input file",
                              layout = Layout(width='100%'))

        display(self.input)

        # Display output formats
        formats = get_formats()
        # Sort by Long Name
        formats.sort_values(by=['long_name'], inplace = True)

        options = list(formats.long_name + " | " + 
                       formats.short_name + " | " +
                       formats.extension)

        self.format = widgets.Dropdown(
                          options = options,
                          value = options[29], # Default GeoTiff format
                          description = "Output format",
                          tooltip = "Long name | GDAL driver | extension",
                          layout=Layout(width='100%'))

        display(self.format)

        # Create output button
        self.output_button = widgets.Button(
            description = 'Select output file',
            tooltip = 'Select output file name and location'
        )

        self.output_button.on_click(self.__on_output_button_clicked)
        display(self.output_button)

    def import_export(self):
        """
        Import and exports file to different GDAL formats
        """
        self.input_button.on_click(self.__on_input_button_clicked)
        display(self.input_button)

    def __clear_cell(self):
        clear_output()

        if self.output_button is not None:
            self.output_button.close()
            self.output_button = None

        if self.translate_button is not None:
            self.translate_button.close()
            self.translate_button = None

        if self.input is not None:
            self.input.close()
            self.input = None

        if self.output is not None:
            self.output.close()
            self.output = None

        if self.format is not None:
            self.format.close()
            self.format = None

