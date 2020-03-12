
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.input_output.utils import *
from TATSSI.notebooks.helpers.utils import *
from TATSSI.qa.EOS.catalogue import Catalogue

from TATSSI.download.modis_downloader import get_modis_data, LOG
from TATSSI.download.viirs_downloader import get_viirs_data

import ogr
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot

from TATSSI.UI.helpers.utils import *

class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

class DownloadersUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(DownloadersUI, self).__init__(parent)
        uic.loadUi('downloaders.ui', self)
        self.parent = parent

        # Tile list
        tiles = self.get_tiles_list()
        self.tiles.clear()
        self.tiles.addItems(tiles)

        # Create TATSSI catalogue object
        catalogue = Catalogue()
        self.fill_products_table(catalogue)

        # Connect methods with events
        self.pbOutputDirectory.clicked.connect(
                self.on_pbOutputDirectory_click)
        self.tvProducts.clicked.connect(
                self.on_tvProducts_click)
        self.pbDownload.clicked.connect(
                self.on_pbDownload_click)

        self.show()

    def fill_products_table(self, catalogue):
        """
        Fills QTableWidget with TATSSI products
        """
        rows, cols = catalogue.products.shape
        # Set number of entries
        self.tvProducts.setRowCount(rows)
        self.tvProducts.setColumnCount(cols)

        for row in range(rows):
            for col in range(cols):
                # Insert item on products TableView
                item = catalogue.products.iloc[row, col]
                self.tvProducts.setItem(row, col,
                        QtWidgets.QTableWidgetItem(item))

        self.tvProducts.resizeColumnsToContents()

        # Set first product as default
        default_product = self.tvProducts.item(0, cols-1).text()
        self.lblProductVersion.setText(default_product)

        # Set default date display format
        self.start_date.setDisplayFormat('dd-MM-yyyy')
        self.end_date.setDisplayFormat('dd-MM-yyyy')

    @pyqtSlot()
    def on_pbDownload_click(self):
        """
        Downloads data based on user selection
        """
        # Get output dir
        output = self.lblOutputDirectory.text()

        # Get product
        product = self.lblProductVersion.text()
        if 'VNP' in product:
            platform = 'VIIRS'
            donwloader = get_viirs_data
        else:
            platform = self.get_modis_platform(product)
            donwloader = get_modis_data

        url, username, password = read_config()

        # Tile
        tile = self.tiles.currentText()

        # Dates needs to be datetime objects
        start_date = datetime.strptime(self.start_date.text(), '%d-%m-%Y')
        end_date = datetime.strptime(self.end_date.text(), '%d-%m-%Y')

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        # Setup the logger
        #from IPython import embed ; ipshell = embed()
        #self.logTextBox.setFormatter(donwloader.LOG

        # Run the downloader
        donwloader(platform = platform,
                   product = product,
                   tiles = tile,
                   output_dir = output,
                   start_date = start_date,
                   end_date = end_date,
                   n_threads = 1,
                   username = username,
                   password = password)

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    @pyqtSlot()
    def on_tvProducts_click(self):
        """
        Sets the lblProductVersion text according to the user's
        selection
        """
        # Get current row
        row = self.tvProducts.currentRow()
        # Get number of columns
        col = self.tvProducts.columnCount() - 1

        product_version = self.tvProducts.item(row, col).text()
        self.lblProductVersion.setText(product_version)

    @pyqtSlot()
    def on_pbOutputDirectory_click(self):
        """
        Opens dialog to select output dir and sets the
        OutputDirectory label text
        """
        output_dir = open_file_dialog('directory')
        self.lblOutputDirectory.setText(output_dir)

    @staticmethod
    def get_tiles_list():
        """
        Gets a list of available MODIS/VIIRS tiles
        """
        current_dir = os.path.join(os.path.dirname(__file__))
        fname = os.path.join(current_dir,
                "../../data/kmz/modis_sin.kmz")
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

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DownloadersUI()
    app.exec_()
