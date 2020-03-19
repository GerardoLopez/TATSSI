
import os
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.input_output.utils import *
from TATSSI.UI.helpers.utils import *

# TATSSI UI dialogs
from TATSSI.UI.downloaders import DownloadersUI
from TATSSI.UI.time_series_generator import TimeSeriesGeneratorUI
from TATSSI.UI.qa_analytics import QAAnalyticsUI
from TATSSI.UI.time_series_smoothing import TimeSeriesSmoothingUI

class TATSSI_UI(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(TATSSI_UI, self).__init__()
        uic.loadUi('tatssi.ui', self)

        # Connect actions
        self.actionDownloaders.triggered.connect(self._downloaders)
        self.actionGenerator.triggered.connect(self._time_series_generator)
        self.actionAnalytics.triggered.connect(self._analytics)
        self.actionSmoothing.triggered.connect(self._time_series_smoothing)

        self.show()

    def _downloaders(self):
        """
        Launch the downloaders dialog
        """
        self.downloaders = DownloadersUI(parent=self)

    def _time_series_generator(self):
        """
        Launch the Time Series Generator dialog
        """
        self.time_series_generator = TimeSeriesGeneratorUI(parent=self)

    def _time_series_smoothing(self):
        """
        For a user selected file open the Time Series Interpolation
        dialog
        """
        fname = open_file_dialog('open')
        if not fname == '':
            self.time_series_smoothing = TimeSeriesSmoothingUI(fname=fname)

    def _analytics(self):
        """
        Launch the QA Analytics dialog
        """
        self.analytics = QAAnalyticsUI(parent=self) 

if __name__ == "__main__":
    app = QApplication([])
    window = TATSSI_UI()
    app.exec_()
