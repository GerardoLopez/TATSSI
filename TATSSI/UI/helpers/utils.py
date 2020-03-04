import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

import json

from pathlib import Path
current_dir = os.path.dirname(__file__)
TATSSI_HOMEDIR = str(Path(current_dir).parents[2])

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
    if dialog_type == 'open':
        fname = QFileDialog.getOpenFileName(parent=None,
                caption="Select a file...",
                directory=TATSSI_HOMEDIR,
                filter="All files (*)")

    elif dialog_type == 'open_specific':
        fname = QFileDialog.getOpenFileName(parent=None,
                caption="Select a file...",
                directory=TATSSI_HOMEDIR,
                filter="%s Files (*.%s)" % \
                        (data_format, extension))

    elif dialog_type == 'save':
        # Get format and extension
        fname = QFileDialog.getSaveFileName(parent=None,
                caption="Save file as...",
                directory=TATSSI_HOMEDIR,
                filter="%s Files (*.%s)" % \
                        (data_format, extension))

    elif dialog_type == 'directory':
        dirname = QFileDialog.getExistingDirectory(parent=None,
                caption="Select a directory...",
                directory=TATSSI_HOMEDIR,
                options=QFileDialog.ShowDirsOnly)

        return dirname

    return str(fname[0])
