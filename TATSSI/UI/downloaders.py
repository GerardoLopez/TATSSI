
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

from TATSSI.download.modis_downloader import get_modis_data
from TATSSI.download.viirs_downloader import get_viirs_data

import ogr

from PyQt5 import QtCore, QtGui, QtWidgets, uic

class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('downloaders_ui.ui', self)

        # Tile list
        tiles = self.get_tiles_list()
        self.tiles.clear()
        self.tiles.addItems(tiles)

        # Create TATSSI catalogue object
        catalogue = Catalogue()
        self.fill_products_table(catalogue)

        self.show()

    def fill_products_table(self, catalogue):
        """
        Fills QTableWidget with TATSSI products
        """
        rows, cols = catalogue.products.shape
        # Set number of entries
        self.products.setRowCount(rows)
        self.products.setColumnCount(cols)

        for row in range(rows):
            for col in range(cols):
                # Insert item
                item = catalogue.products.iloc[row, col]
                self.products.setItem(row, col, QtWidgets.QTableWidgetItem(item))

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

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
