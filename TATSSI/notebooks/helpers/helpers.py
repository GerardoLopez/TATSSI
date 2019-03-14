
import os
import sys

# TATSSI modules
sys.path.append ("/home/glopez/Projects/TATSSI")
from TATSSI.input_output.translate import Translate
from TATSSI.input_output.utils import *
from TATSSI.qa import catalogue
from TATSSI.download import modis_downloader

from IPython.display import clear_output
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual

from ipywidgets import Layout
from ipywidgets import Button, HBox, VBox

import ipywidgets as widgets

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

import gdal
import pandas as pd

from beakerx import TableDisplay

from datetime import datetime

def OpenFileDialog(dialog_type = 'open',
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

    return str(fname[0])

class Download():
    def __init__(self):
        """
        Class to handle donwload operations within the notebook
        """
        # Create TATSSI catalogue object
        self.catalogue = catalogue.Catalogue()

        self.select_product_button = widgets.Button(
                layout = Layout(width='99.6%'),
                description = 'Select product and dates',
                tooltip = 'Select product to be downloaded ' + \
                      'and required dates.')

        self.product = None
        self.product_table = None

        self.download()

    def __on_product_button_button_clicked(self, b):
        """
        Shows table with available products
        """
        def on_product_table_double_click(row, col, table):
            """Internal function to update product and dates"""
            value = table.values[row][-1]
            self.product.value = f"<b>{value}</b>"
            # Update dates
            start = self.__string_to_datetime(table.values[row][-5])
            self.start_date.value = start
            end = self.__string_to_datetime(table.values[row][-6])
            self.end_date.value = end

        self.__clear_cell()
        # Display button
        display(self.select_product_button)

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
                placeholder = "Product to download",
                description = "Product and version:",
                layout = Layout(width = '100%'),
                style = style)

        display(self.product)

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

    def download(self):
        """
        Downloads a product from the LPDAAC
        """
        self.select_product_button.on_click(
                self.__on_product_button_button_clicked)
        # Display button
        display(self.select_product_button)

    def __clear_cell(self):
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

        target_img = OpenFileDialog('save', data_format, extension)
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

        source_img = OpenFileDialog('open')
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

