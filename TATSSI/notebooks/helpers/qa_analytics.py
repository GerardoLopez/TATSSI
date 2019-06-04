
import os
import sys

# TATSSI modules
HomeDir = os.path.join(os.path.expanduser('~'))
SrcDir = os.path.join(HomeDir, 'Projects', 'TATSSI')
sys.path.append(SrcDir)

from TATSSI.time_series.generator import Generator
from TATSSI.input_output.utils import *
from TATSSI.qa.EOS.catalogue import Catalogue

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
import numpy as np

from rasterio import logging as rio_logging
from datetime import datetime

import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

class QAnalytics():
    """
    Class to provide QA analysis tools
    """
    def __init__(self, product, version):
        # Create TATSSI catalogue object
        self.catalogue = Catalogue()

        # Time series object
        # set on __load_time_series
        self.ts = None

        # All QA definitions
        # set on __get_qa_defs
        self.qa_defs = None

        # QA definition to analise
        # set on qa_ui
        self.qa_def = None

        # User's QA selection
        # set on qa_ui
        self.user_qa_selection = None

        # Display QA User Interface
        self.__qa_ui()

    def __qa_ui():
        """
        QA user interface
        """
        self.qa_def = self.qa_defs[1]

        qa_flags = self.qa_def.Name.unique()
        qa_layer = self.qa_def.QualityLayer.unique()

        qa_layer_header = HTML(
            value = f"<b>{qa_layer[0]}</b>",
            description='QA layer:'
        )
        display(qa_layer_header)

        self.user_qa_selection = dict((element, '') for element in qa_flags)
        # Fill default selection
        for i, selection in enumerate(self.user_qa_selection):
            self.user_qa_selection[selection] = tuple(
                [self.qa_def[self.qa_def.Name == selection].Description.tolist()[0]]
            )

        qa_flag = Select(
            options=qa_flags,
            value=qa_flags[0],
            rows=len(qa_flags),
            description='QA Parameter name:',
            style = {'description_width': 'initial'},
            layout={'width': '350px'},
            disabled=False
        )

        def on_qa_flag_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
            qa_flag_value = change.owner.value
        
            # Get user selection before changing qa description
            tmp_selection = self.user_qa_selection[qa_flag_value]

            _options = self.qa_def[self.qa_def.Name == qa_flag_value].Description.tolist()
            qa_description.options = _options
        
            qa_description.rows = len(_options)
            qa_description.value = tmp_selection
    
        qa_flag.observe(on_qa_flag_change)

        qa_description = SelectMultiple(
            options=tuple(
                self.qa_def[self.qa_def.Name == qa_flag.value].Description.tolist()
            ),
            value=tuple(
                [self.qa_def[self.qa_def.Name == qa_flag.value].Description.tolist()[0]]
            ),
            rows=len(self.qa_def[self.qa_def.Name == qa_flag.value].Description.tolist()),
            description='Description',
            disabled=False,
            style = {'description_width': 'initial'},
            layout={'width': '400px'}
        )

        def on_qa_description_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
            self.user_qa_selection[qa_flag.value] = qa_description.value

        qa_description.observe(on_qa_description_change)

        def select_all_qa(b):
            for i, selection in enumerate(self.user_qa_selection):
                self.user_qa_selection[selection] = tuple(
                    self.qa_def[self.qa_def.Name == selection].Description.tolist()
                )
    
            qa_flag.value = qa_flags[0]
            qa_description.value = self.user_qa_selection[qa_flags[0]]

        # Select all button
        select_all = Button(
            description = 'Select ALL',
            layout={'width': '20%'}
        )

        select_all.on_click(select_all_qa)

        # Default selection
        select_default = Button(
            description = 'Default selection',
            layout={'width': '20%'}
        )

        def select_default_qa(b):
            # Fill default selection
            for i, selection in enumerate(self.user_qa_selection):
                self.user_qa_selection[selection] = tuple(
                    [self.qa_def[self.qa_def.Name == selection].Description.tolist()[0]]
                )
    
            qa_flag.value = qa_flags[0]
            qa_description.value = self.user_qa_selection[qa_flags[0]]

        select_default.on_click(select_default_qa)

        left_box = VBox([qa_flag])
        right_box = VBox([qa_description])
        HBox([qa_flag, right_box, select_all, select_default],
             layout={'height': '350px',
                     'width' : '99%'}
        )

    def __load_time_series(source_dir, product):
        """
        Loads existing time series using the TATSSI
        time series Generator class
        :param source_dir: root directory where GeoTiff's and VRTs
                           are stored
        :param product: product and version, e.g. 'MOD13A2.006'
        :return time series TATSSI object
        """
        # Create time series generator object
        tsg = Generator(source_dir = DataDir,
                        product = product)

        # Load time series
        self.ts = tsg.load_time_series()

        # User's QA selection
        self.user_qa_selection

    def __get_qa_defs(product, version):
        """
        Get QA definitions for a particular product and version and
        changes the 'Value' field from decimal to binary. This is 
        neccesary because the decoded QA GeoTifts have this binary
        values stored.
        """
        qa_defs = self.catalogue.get_qa_definition(product=product,
                                                   version=version)
        # Convert to binary
        for qa_def in qa_defs:
            binary_vals_list = []
            for qa_value in qa_def.Value:
                binary_vals_list.append(int(bin(qa_value)[2:]))

            # Update the 'Value' field
            qa_def['Value'] = binary_vals_list

        self.qa_defs = qa_defs


