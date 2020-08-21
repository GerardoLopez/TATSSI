
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import urllib
import osr

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

        # Connect event
        self.pbGetExtent.clicked.connect(
                self.on_pbGetExtent_click)

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
            proj = None

        fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                subplot_kw=dict(projection=proj))

        # Set axis as attribute to access extent
        self.ax = ax
        self.extent = ax.get_extent()

        if proj is not None:
            try:
                cfeature.BORDERS.geometries()
                ax.coastlines(resolution='50m', color='black')
                ax.add_feature(cfeature.BORDERS, edgecolor='black')
            except urllib.error.URLError:
                pass
            ax.gridlines()

        # Make all values NaN
        # data_array = (data_array.astype(np.float32) * np.nan)
        # Plot
        data_array[0].plot.imshow(
                ax=ax, add_colorbar=False, cmap='viridis',
                transform=proj
        )

        ax.set_frame_on(False)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.title.set_text('')

        # Set plot on the plot widget
        self.plotWidget = FigureCanvas(fig)
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
        fig.canvas.draw()
        plt.tight_layout()

