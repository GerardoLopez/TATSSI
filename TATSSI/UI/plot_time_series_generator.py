
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.UI.helpers.utils import *

from osgeo import ogr
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader as cReader
from cartopy.feature import ShapelyFeature
from osgeo import osr

import urllib

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
        as NavigationToolbar

from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont

def get_projection(proj4_string):
    """
    Get spatial reference system from PROJ4 string
    """
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4_string)

    return srs

class PlotExtent(QtWidgets.QMainWindow):
    """
    Plot standard anomalies from a TATSSI climatogy for a particular year
    """
    def __init__(self, parent=None):
        super(PlotExtent, self).__init__(parent)

        uic.loadUi('plot_time_series_generator.ui', self)
        self.parent = parent

        # Extent
        self.extent = None
        # Figure and axis
        self.fig, self.ax = None, None

        # Connect events
        self.pbGetExtent.clicked.connect(
                self.on_pbGetExtent_click)

        self.pbOverlay.clicked.connect(
                self.on_pbOverlay_click)

    def on_pbOverlay_click(self):
        """
        EXPERIMENTAL
        Overlay a specific geometry on maps
        """
        fname = open_file_dialog(dialog_type = 'open_specific',
                data_format = 'Shapefile',
                extension = 'shp')

        # If there is no selection
        if fname == '':
            return None

        # If file does not exists
        if os.path.exists(fname) is False:
            return None

        # Open file
        spatial_reference = self.get_shapefile_spatial_reference(fname)

        # Get ellipsoid/datum parameters
        globe=ccrs.Globe(ellipse=None,
                semimajor_axis=spatial_reference.GetSemiMajor(),
                semiminor_axis=spatial_reference.GetSemiMinor())

        self.shapefile_projection = ccrs.Sinusoidal(globe=globe)
            #ccrs.CRS(spatial_reference.ExportToProj4())

        try:
            shape_feature = ShapelyFeature(cReader(fname).geometries(),
                self.shapefile_projection, facecolor='none')

            for _axis in [self.ax]:
                _axis.add_feature(shape_feature,
                        edgecolor='gray')
        except:
            return None

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_pbGetExtent_click(self):
        """
        Set self.extent based on user selection
        """
        self.extent = self.ax.get_extent()

    def _plot(self, data_array):
        """
        From the TATSSI Time Series Generator object plots the
        extent of data with coastline and political division
        """
        # Get projection from first data variable
        proj4_string = data_array.crs

        # If projection is Sinusoidal
        srs = get_projection(proj4_string)
        if srs.GetAttrValue('PROJECTION') == 'Sinusoidal':
            globe=ccrs.Globe(ellipse=None,
                semimajor_axis=6371007.181,
                semiminor_axis=6371007.181)

            proj = ccrs.Sinusoidal(globe=globe)
        else:
            globe = ccrs.Globe(ellipse='WGS84')
            self.projection = ccrs.Mollweide(globe=globe)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8),
                subplot_kw=dict(projection=proj))

        self.extent = self.ax.get_extent()

        if proj is not None:
            try:
                cfeature.BORDERS.geometries()
                self.ax.coastlines(resolution='50m', color='black')
                self.ax.add_feature(cfeature.BORDERS, edgecolor='black')
            except urllib.error.URLError:
                pass

            # Gridlines
            self.ax.gridlines()

        # Make all values NaN
        # data_array = (data_array.astype(np.float32) * np.nan)
        # Plot
        data_array[0].plot.imshow(
                ax=self.ax, add_colorbar=False, cmap='viridis',
                transform=proj
        )

        self.ax.set_frame_on(False)
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.ax.title.set_text('')

        # Set plot on the plot widget
        self.plotWidget = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self.content_plot)
        lay.setContentsMargins(0, 40, 0, 0)
        lay.addWidget(self.plotWidget)

        # Add toolbar
        font = QFont()
        font.setPointSize(10)

        toolbar = NavigationToolbar(self.plotWidget, self)
        toolbar.setFont(font)

        self.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

        # Needed in order to use a tight layout with Cartopy axes
        self.fig.canvas.draw()
        plt.tight_layout()

    @staticmethod
    def get_shapefile_spatial_reference(fname):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataset = driver.Open(fname)

        layer = dataset.GetLayer()
        spatialRef = layer.GetSpatialRef()

        return spatialRef

