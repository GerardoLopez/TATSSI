
import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

import json

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(__file__)
src_dir = Path(current_dir).parents[2]
sys.path.append(str(src_dir.absolute()))

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
                    "Select a file...", './',
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
