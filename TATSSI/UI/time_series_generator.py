
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.input_output.utils import has_subdatasets, \
        get_subdatasets, get_formats
from TATSSI.input_output.translate import Translate
#from TATSSI.notebooks.helpers.utils import *
from TATSSI.qa.EOS.catalogue import Catalogue
from TATSSI.time_series.generator import Generator
from TATSSI.UI.plot_time_series_generator import PlotExtent

from osgeo import gdal
import xarray as xr
from glob import glob

from osgeo import ogr
from datetime import datetime
import tempfile

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot

from TATSSI.UI.helpers.utils import *

class TimeSeriesGeneratorUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TimeSeriesGeneratorUI, self).__init__(parent)
        uic.loadUi('time_series_generator.ui', self)
        self.parent = parent

        # Create TATSSI catalogue object
        catalogue = Catalogue()
        self.fill_products_table(catalogue)

        # Connect methods with events
        self.pbDataDirectory.clicked.connect(
                self.on_pbDataDirectory_click)
        self.tvProducts.clicked.connect(
                self.on_tvProducts_click)
        self.pbGenerateTimeSeries.clicked.connect(
                self.on_pbGenerateTimeSeries_click)
        self.pbGetSubsetExtent.clicked.connect(
                self.on_pbGetSubsetExtent_click)

        # Disable pbPlotExtent
        self.pbGetSubsetExtent.setEnabled(False)

        # Data formats
        self.formats = None
        self._fill_formats()
        self.data_format = None

        # List of dialogs
        self.dialogs = list()

        self.show()

    def _fill_formats(self):
        """
        Fill formats combo box
        """
        self.formats = get_formats()
        # Sort by Long Name
        # self.formats.sort_values(by=['long_name'], inplace = True)

        _formats_list = list(self.formats.long_name + " | " +
                             self.formats.short_name + " | " +
                             self.formats.extension)

        self.cmbFormats.addItems(_formats_list)
        self.cmbFormats.setEnabled(False)

    @pyqtSlot()
    def on_pbGetSubsetExtent_click(self):
        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        dialog = PlotExtent(self)
        self.dialogs = [dialog]

        data = self.__get_data_from_first_file()
        if data is None:
            return

        dialog._plot(data)

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

        dialog.show()

        # Change font style in pbGetSubsetExtent button
        _font = QtGui.QFont()
        _font.setBold(True)
        self.pbGetSubsetExtent.setFont(_font)

    def __get_data_from_first_file(self):
        """
        Get a xarray DataArray from the first file in the data directory
        :return data: xarray DataArray
        """
        # Get data directory
        data_dir = self.lblDataDir.text()

        # Get first file name
        fname = glob(os.path.join(data_dir, '*'))[0]

        if len(fname) == 0:
            nl = '\n'
            msg = f'{data_dir} does not contain files.\n'
            _dialog = QtWidgets.QMessageBox()
            _dialog.setIcon(QtWidgets.QMessageBox.Warning)
            _dialog.about(self, 'TATSSI Warning', msg)
            return None

        if has_subdatasets(fname)[0] == True:
            sd = get_subdatasets(fname)

            tmp_fname = f'{tempfile.NamedTemporaryFile().name}.tif'
            Translate(source_img=sd[0][0],
                    target_img=tmp_fname)

            data = xr.open_rasterio(tmp_fname)
        else:
            data = xr.open_rasterio(fname)

        # Delete tmp file
        try:
            os.remove(tmp_fname)
        except:
            pass

        return data

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

    @pyqtSlot()
    def on_pbGenerateTimeSeries_click(self):
        """
        Generate time series for data located on user-selected
        data directory for a specific product and version
        """
        # Set extent
        if len(self.dialogs) == 1:
            extent = self.dialogs[0].extent
        else:
            extent = None

        # Get output dir
        data_dir = self.lblDataDir.text()

        # Get product
        product, version = self.lblProductVersion.text().split('.')

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        # Enable progress bar
        self.progressBar.setEnabled(True)
        self.progressBar.setValue(0)

        # Create the time series generator object
        tsg = Generator(source_dir=self.lblDataDir.text(),
                        product=product, version=version,
                        data_format=self.data_format,
                        progressBar=self.progressBar,
                        extent=extent)

        # Generate the time series
        tsg.generate_time_series()

        # Disable progress bar
        self.progressBar.setValue(0)
        self.progressBar.setEnabled(False)

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
    def on_pbDataDirectory_click(self):
        """
        Opens dialog to select output dir and sets the
        OutputDirectory label text
        """
        data_dir = open_file_dialog('directory')
        if len(data_dir) == 0:
            return

        # Check if data_dir has sub directories
        sub_dirs = []
        data_format = []
        for sub_dir in os.listdir(data_dir):
            sub_dir = os.path.join(data_dir, sub_dir)
            if os.path.isdir(sub_dir) is True:
                sub_dirs.append(sub_dir)
            else:
                fname, fextension = os.path.splitext(sub_dir)
                data_format.append(fextension)

        if len(sub_dirs) > 0:
            nl = '\n'
            msg = (f'{data_dir} contains sub-directories.\n'
                   f'\nTATSSI time series generates sub-directories '
                   f'per each variable/subdataset.\n'
                   f'\nRemove ALL sub-directories first.')
            _dialog = QtWidgets.QMessageBox()
            _dialog.setIcon(QtWidgets.QMessageBox.Warning)
            _dialog.about(self, 'TATSSI Warning', msg)

            return None
        else:
            _format = self.formats[self.formats.extension.str.contains(data_format[0][1::])]
            if len(_format) == 0:
                nl = '\n'
                msg = (f'{data_dir} contains unsupported files.\n'
                       f'\nTATSSI uses GDAL to process files. '
                       f'Ensure datasets are supported by GDAL.\n')
                _dialog = QtWidgets.QMessageBox()
                _dialog.setIcon(QtWidgets.QMessageBox.Warning)
                _dialog.about(self, 'TATSSI Warning', msg)

                return None
            else:
                self.cmbFormats.setCurrentIndex(_format.index[0])
              
                self.data_format = data_format[0][1::]
                self.lblDataDir.setText(data_dir)
                # Enable get subset extent button
                self.pbGetSubsetExtent.setEnabled(True)
                # Enable combo formats
                self.cmbFormats.setEnabled(True)

    @staticmethod
    def all_same(items):
        """
        Check if all elements in list are the same
        """
        return all(x == items[0] for x in items)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = TimeSeriesGeneratorUI()
    app.exec_()
