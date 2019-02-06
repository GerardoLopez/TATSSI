
import os
import sys
sys.path.append ("/home/glopez/Projects/TATSSI")
from TATSSI.input_output.translate import translate

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

class ImportExport():

    def __init__(self):
        """
        Class to handle Input/Output operations in within the notebook
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

    def OpenFileDialog(self, dialog_type = 'open',
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
            data_format = self.format.value.split('|')[0].strip()
            extension = self.format.value.split('|')[2].strip()

            fname = QFileDialog.getSaveFileName(None,
                        "Save file as...", '.',
                        filter="%s Files (*.%s)" % \
                                (data_format, extension))

        return str(fname[0])

    def on_output_button_clicked(self, b):
        """
        Based on user file selection displays the output file
        """
        target_img = self.OpenFileDialog('save')
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

        self.translate_button.on_click(self.on_translate_button_clicked)
        display(self.translate_button)

    def on_translate_button_clicked(self, b):
        """
        Performs the translation into an output file with selected format
        """
        # Checks...
        try:
            # Use GDAL exceptions
            gdal.UseExceptions()
            tmp_d = gdal.Open(self.input.value)
        except Exception as e:
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
        translate(self.input.value, self.output.value, driver)

    def on_input_button_clicked(self, b):
        """
        Based on user file selection displays either the input file
        or the SubDatasets of the selected file
        """
        self.clear_cell()
        display(self.input_button)

        source_img = self.OpenFileDialog('open')
        if len(source_img) == 0:
            # If there's no source file, do nothing...
            return None

        if translate.has_subdatasets(source_img):
            # Get SubDatasets
            sds = translate.get_subdatasets(source_img)
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
        formats = translate.get_formats()
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

        self.output_button.on_click(self.on_output_button_clicked)
        display(self.output_button)

    def import_export(self):
        """
        Import and exports file to different GDAL formats
        """
        self.input_button.on_click(self.on_input_button_clicked)
        display(self.input_button)

    def clear_cell(self):
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

