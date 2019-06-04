
import os
import sys

# TATSSI modules
HomeDir = os.path.join(os.path.expanduser('~'))
SrcDir = os.path.join(HomeDir, 'Projects', 'TATSSI')
sys.path.append(SrcDir)

from TATSSI.input_output.utils import *
from TATSSI.notebooks.helpers.utils import *
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

class Download():
    """
    Class to handle donwload operations within the notebook
    """
    def __init__(self):
        # Create TATSSI catalogue object
        self.catalogue = Catalogue()

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
