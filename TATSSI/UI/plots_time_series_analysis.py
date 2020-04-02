
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
        as NavigationToolbar

from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot

class PlotAnomalies(QtWidgets.QMainWindow):
    """
    Plot standard anomalies from a TATSSI climatogy for a particular year
    """
    def __init__(self, parent=None):
        super(PlotAnomalies, self).__init__(parent)

    def _plot(self, anomalies, year, cmap='BrBG', dpi=72):
        """
        From the TATSSI Time Series Analysis object plots the
        standard anomalies
        """
        uic.loadUi('plot.ui', self)

        a_plot = anomalies.plot.imshow(col='time', robust=True,
                col_wrap=6, cmap=cmap, aspect=1, size=2)

        for _ax in a_plot.axes.ravel():
            _ax.set_aspect('equal')
    
        # Set plot on the plot widget
        self.plotWidget = FigureCanvas(a_plot.fig)
        lay = QtWidgets.QVBoxLayout(self.content_plot)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)
        # Add toolbar
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                NavigationToolbar(self.plotWidget, self))
