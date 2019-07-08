
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(__file__)
src_dir = Path(current_dir).parents[2]
sys.path.append(str(src_dir.absolute()))

from TATSSI.input_output.translate import Translate
from TATSSI.notebooks.helpers.utils import *
from TATSSI.input_output.utils import *

# Widgets
import ipywidgets as widgets
from ipywidgets import Layout
from ipywidgets import Button, HBox, VBox
from ipywidgets import interact, interactive, fixed, interact_manual

from IPython.display import clear_output
from IPython.display import display

import gdal, ogr
import pandas as pd
import xarray as xr
from rasterio import logging as rio_logging
from datetime import datetime

import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

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

