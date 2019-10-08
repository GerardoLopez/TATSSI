
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(__file__)
src_dir = Path(current_dir).parents[2]
sys.path.append(str(src_dir.absolute()))

from TATSSI.time_series.generator import Generator
from TATSSI.input_output.utils import *
from TATSSI.qa.EOS.catalogue import Catalogue

# Widgets
import ipywidgets as widgets
from ipywidgets import Layout
from ipywidgets import Select, SelectMultiple
from ipywidgets import Button, HBox, VBox, HTML, IntProgress
#from ipywidgets import interact, interactive, fixed, interact_manual

from beakerx import TableDisplay

from IPython.display import clear_output
from IPython.display import display

import json
import collections
from itertools import groupby as i_groupby
import gdal, ogr
import pandas as pd
import xarray as xr
import numpy as np

from rasterio import logging as rio_logging
from datetime import datetime

import matplotlib
matplotlib.use('nbAgg')
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class Analytics():
    """
    Class to provide QA analytics
    """
    def __init__(self, source_dir, product, version,
                 year=None, start=None, end=None):

        # Check input parameters
        if os.path.exists(source_dir) is True:
            self.source_dir = source_dir
        else:
            print(f"{source_dir} does not exists!")
            return None

        if isinstance(product, str) and len(product) > 3:
            self.product = product
        else:
            return None

        if isinstance(version, str) and len(version) == 3:
            self.version = version
        else:
            return None

        # QA definition to analise
        # set on qa_ui
        self.qa_def = None

        # User's QA selection
        # set on qa_ui
        self.user_qa_selection = None

        # Mask based on user_qa_selection
        # set on __analytics
        self.mask = None

        # Percentage of data available after masking
        # set on __analytics
        self.pct_data_available = None

        # Max gap length
        # set on get_max_gap_length
        self.max_gap_length = None

        # Create TATSSI catalogue object
        self.catalogue = Catalogue()

        # Time series object
        self.year = year
        self.start, self.end = start, end
        self.ts = self.__load_time_series()

        # All QA definitions
        self.qa_defs = self.__get_qa_defs()

    def plot_analytics(self):
        """
        Plot the percentage of data available and the max gap length
        """
        # Clear cell
        self.__clear_cell()
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9,5),
                                     sharex=True, sharey=True)

        ax.set_frame_on(False)
        bx.set_frame_on(False)

        ax.axis('off')
        bx.axis('off')

        self.pct_data_available.plot(
                ax=ax,
                cbar_kwargs={'orientation':'horizontal',
                             'pad' : 0.01},
        )

        self.max_gap_length.plot(
                ax=bx,
                cbar_kwargs={'orientation':'horizontal',
                             'pad' : 0.01},
        )

        ax.set_aspect('equal')
        ax.title.set_text('% of data available')

        bx.set_aspect('equal')
        bx.title.set_text('Max gap-length')

        plt.margins(tight=True)
        plt.tight_layout()
        plt.show()

    def ui(self):
        """
        QA user interface
        """
        # Clear cell
        self.__clear_cell()

        if self.qa_def is None:
            # Use the first or unique QA
            self.qa_def = self.qa_defs[0]

        qa_flags = self.qa_def.Name.unique()
        qa_layer = self.qa_def.QualityLayer.unique()

        qa_layer_header = HTML(
            value = f"<b>{qa_layer[0]}</b>",
            description='QA layer:'
        )

        self.user_qa_selection = collections.OrderedDict(
                (element, '') for element in qa_flags)

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
            layout={'width': '400px'},
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
            layout={'width': '19%'}
        )

        select_all.on_click(select_all_qa)

        # Default selection
        select_default = Button(
            description = 'Default selection',
            layout={'width': '19%'}
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
        #_HBox = HBox([qa_flag, right_box, select_all, select_default],
        _HBox_qa = HBox([left_box, right_box],
                        layout={'height': '300px',
                                'width' : '99%'}
        )

        analytics = Button(
            description = 'QA analytics',
            layout={'width': '19%'}
        )
        analytics.on_click(self.__analytics)

        analytics_settings_save = Button(
            description = 'Save QA analytics',
            layout={'width': '19%'}
        )
        analytics_settings_save.on_click(self.__analytics_settings_save)

        # Load user-defined settings
        analytics_settings_load = Button(
            description = 'Load QA analytics',
            layout={'width': '19%'}
        )

        def __analytics_settings_load(b):
            # Load user-defined QA saved settings from a JSON file
            fname = self.qa_def.QualityLayer.unique()[0]
            fname = f"{fname}.json"
            fname = os.path.join(self.source_dir, fname)

            if os.path.exists(fname) is False:
                pass

            with open(fname, 'r') as f:
                self.user_qa_selection = collections.OrderedDict(
                        json.loads(f.read()))

            qa_flag.value = qa_flags[0]
            qa_description.value = self.user_qa_selection[qa_flags[0]]

        analytics_settings_load.on_click(__analytics_settings_load)

        # Display QA HBox
        display(qa_layer_header, _HBox_qa)
        
        _HBox_buttons = HBox([select_all, select_default, analytics,
                              analytics_settings_save,
                              analytics_settings_load])

        display(_HBox_buttons)

    def __load_time_series(self):
        """
        Loads existing time series using the TATSSI
        time series Generator class
        :attr self.source_dir: root directory where GeoTiff's and VRTs
                           are stored
        :attr self.product: product name, e.g. 'MOD13A2'
        :atte self.version: version of the product '006'
        :return time series TATSSI object
        """
        # Create time series generator object
        tsg = Generator(source_dir=self.source_dir,
                product=self.product, version=self.version,
                year=self.year, start=self.start, end=self.end)

        # Load time series
        return tsg.load_time_series()

    def __get_qa_defs(self):
        """
        Get QA definitions for a particular product and version and
        changes the 'Value' field from decimal to binary. This is 
        neccesary because the decoded QA GeoTifts have this binary
        values stored.
        :attr self.product: product name, e.g. 'MOD13A2'
        :atte self.version: version of the product '006'
        :return QA definitions DataFrame
        """
        qa_defs = self.catalogue.get_qa_definition(product=self.product,
                                                   version=self.version)
        # Convert to binary
        for qa_def in qa_defs:
            binary_vals_list = []
            for qa_value in qa_def.Value:
                binary_vals_list.append(int(bin(qa_value)[2:]))

            # Update the 'Value' field
            qa_def['Value'] = binary_vals_list

        return qa_defs

    def __analytics_settings_save(self, b):
        """
        Save the current user-defined QA setting into a JSON file
        """
        fname = self.qa_def.QualityLayer.unique()[0]
        fname = f"{fname}.json"
        fname = os.path.join(self.source_dir, fname)

        with open(fname, 'w') as f:
            f.write(json.dumps(self.user_qa_selection))

        LOG.info(f"QA settings file {fname} written to disk.")

    def __analytics_settings_load(self, b):
        """
        Load user-defined QA saved settings from a JSON file
        """
        fname = self.qa_def.QualityLayer.unique()[0]
        fname = f"{fname}.json"
        fname = os.path.join(self.source_dir, fname)

        if os.path.exists(fname) is False:
            pass

        with open(fname, 'r') as f:
            self.user_qa_selection = collections.OrderedDict(
                    json.loads(f.read()))

    def __analytics(self, b):
        """
        Uses the self.user_qa_selection OrderedDictionary to extract
        the corresponding QA values and create a mask of dimensions:
            (number of qa layers, time steps, cols(lat), rows(lon))
        Additionally computes the temporal mask and the max gap length
        """
        progress_bar = IntProgress(
                value=0,
                min=0,
                max=len(self.user_qa_selection),
                step=1,
                description='',
                bar_style='', # 'success', 'info', 'warning', 'danger' or ''
                orientation='horizontal',
                style = {'description_width': 'initial'},
                layout={'width': '50%'}
        )
        display(progress_bar)

        n_qa_layers = len(self.user_qa_selection)

        # Get the name of the first data var to extract its shape
        for k, v in self.ts.data.data_vars.items():
            break

        # Create mask xarray
        _time, _latitude, _longitude = self.ts.data.data_vars[k].shape
        mask = np.zeros((n_qa_layers, _time, _latitude, _longitude),
                        np.int8)

        qa_layer = self.qa_def.QualityLayer.unique()

        # QA layer user to create mask
        _qa_layer = getattr(self.ts.qa, f"qa{qa_layer[0]}")

        for i, user_qa in enumerate(self.user_qa_selection):
            progress_bar.value = i

            user_qa_fieldname = user_qa.replace(" ", "_").replace("/", "_")
            progress_bar.description = "Masking by user QA selection"

            for j, qa_value in enumerate(self.user_qa_selection[user_qa]):
                qa_value_field_name = qa_value.replace(" ", "_")

                qa_flag_val = self.qa_def[(self.qa_def.Name == user_qa) & 
                        (self.qa_def.Description == qa_value)].Value.iloc[0]

                if j == 0 :
                    mask[i] = (_qa_layer[user_qa_fieldname] == qa_flag_val)
                else:
                    mask[i] = np.logical_or(
                                  mask[i], _qa_layer[user_qa_fieldname] == qa_flag_val)

        # Remove progress bar
        progress_bar.close()
        del progress_bar

        self.__temp_mask = mask
        mask = xr.DataArray(np.all(self.__temp_mask, axis=0),
                            coords=[v.time.data,
                                    v.latitude.data,
                                    v.longitude.data],
                            dims=['time', 'latitude', 'longitude'])

        mask.attrs = v.attrs

        self.mask = mask

        # Create the percentage of data available mask
        # Get the per-pixel per-time step binary mask
        pct_data_available = (self.mask.sum(axis=0) * 100.0) / _time
        pct_data_available.latitude.data = v.latitude.data
        pct_data_available.longitude.data = v.longitude.data
        # Set the pct_data_available object
        self.pct_data_available = pct_data_available

        # Using the computed mask get the max gap length
        self.__get_max_gap_length()

    def __get_max_gap_length(self):
        """
        Compute the max gep length of a masked time series
        """
        # TODO
        # This function should be paralelised! 

        bands, rows, cols = self.mask.shape
        max_gap_length = np.zeros((rows, cols), np.int16)

        progress_bar = IntProgress(
                value=0,
                min=0,
                max=10,
                step=1,
                description='Computing max gap length...',
                bar_style='', # 'success', 'info', 'warning', 'danger' or ''
                orientation='horizontal',
                style = {'description_width': 'initial'},
                layout={'width': '50%'}
        )
        display(progress_bar)

        for i in range(rows):
            progress_bar.value = int((i*10.)/rows)
            for j in range(cols):
                for key, group in i_groupby(self.mask.data[:,i,j]):
                    if key == False:
                        _gap_lenght = len(list(group))
                        if _gap_lenght > 0 and _gap_lenght > max_gap_length[i,j]:
                            max_gap_length[i,j] = _gap_lenght

        # Remove progress bar
        progress_bar.close()
        del progress_bar

        # Create xarray DataArray
        _max_gap_length = xr.DataArray(max_gap_length,
                            coords=[self.mask.latitude.data,
                                    self.mask.longitude.data],
                            dims=['latitude', 'longitude'])

        max_gap_length = None

        self.max_gap_length = _max_gap_length

    def __clear_cell(self):
        """ Clear cell """
        clear_output()
