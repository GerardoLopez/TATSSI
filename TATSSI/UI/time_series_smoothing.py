
import os
import sys

import numpy as np

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.time_series.smoothn import smoothn
from TATSSI.time_series.analysis import Analysis
from TATSSI.time_series.smoothing import Smoothing
from TATSSI.notebooks.helpers.time_series_smoothing import \
        TimeSeriesSmoothing

# Smoothing methods
import statsmodels.tsa.api as tsa

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
        as NavigationToolbar

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot

class TimeSeriesSmoothingUI(QtWidgets.QMainWindow):
    def __init__(self, fname, parent=None):
        super(TimeSeriesSmoothingUI, self).__init__(parent)
        uic.loadUi('plot_smoothing.ui', self)
        self.parent = parent

        # Set input file name
        self.fname = fname

        # Plot input data
        self._plot()

        self.show()

    def __set_variables(self):
        """
        Set variables from TATSSI Time Series object
        """
        # TATSSI time series object
        self.ts = Analysis(fname=self.fname)

        # imshow plots
        self.img_imshow = None

        # Get widgets
        self.time_steps.currentIndexChanged.connect(
                self.__on_time_steps_change)

        self.pbSmooth.clicked.connect(
                self.on_pbSmooth_click)

        # Data variables
        self.data_vars.addItems(self.__fill_data_variables())
        # Time steps
        self.time_steps.addItems(self.__fill_time_steps())

        # Create plot objects
        self.__create_plot_objects()

        # Populate plots
        self.__populate_plots()

        # Set smoothing methods
        self.__fill_smoothing_methods()

    def on_pbSmooth_click(self):
        """
        Performs a smoothing for using a specific user selected
        method
        """
        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        # Output file name
        smoothing_method = self.smoothing_methods.selectedItems()[0].text()
        _fname, _ext = os.path.splitext(self.fname)
        output_fname = f'{_fname}.{smoothing_method}.tif'

        smoother = Smoothing(fname=self.fname, output_fname=output_fname,
                smoothing_methods=[smoothing_method])

        smoother.smooth()

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    def __fill_data_variables(self):
        """
        Fill the data variables dropdown list
        """
        data_vars = []
        for data_var in self.ts.data.data_vars:
            data_vars.append(data_var)

        return data_vars

    def __fill_time_steps(self):
        """
        Fill the time steps dropdown list
        """
        tmp_ds = getattr(self.ts.data, self.data_vars.currentText())

        time_steps = np.datetime_as_string(tmp_ds.time.data, 'm').tolist()

        return time_steps

    def __fill_smoothing_methods(self):
        """
        Fill smoothing methods
        """
        smoothing_methods = ['smoothn',
                             'ExponentialSmoothing',
                             'SimpleExpSmoothing',
                             'Holt']

        self.smoothing_methods.addItems(smoothing_methods)

    def on_click(self, event):
        """
        Event handler
        """
        # Event does not apply for time series plot
        # Check if the click was in a
        if event.inaxes in [self.ts_p]:
            return

        # Clear subplot
        self.ts_p.clear()

        # Delete last reference point
        if len(self.img_p.lines) > 0:
            del self.img_p.lines[0]

        # Draw a point as a reference
        self.img_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=7, alpha=0.7)

        # Interpolated data to smooth
        img_plot_sd = self.img_ds.sel(longitude=event.xdata,
                                      latitude=event.ydata,
                                      method='nearest')
        if img_plot_sd.chunks is not None:
            img_plot_sd = img_plot_sd.compute()

        # Plots
        img_plot_sd.plot(ax=self.ts_p, color='black',
                linestyle = '-', linewidth=1, label='Original data')

        # For every smoothing method selected by the user
        for method in self.smoothing_methods.selectedItems():
            method = method.text()

            y = img_plot_sd.data
            s = float(self.smooth_factor.value())

            if method == 'smoothn':
                # Smoothing
                fittedvalues = smoothn(y, isrobust=True,
                        s=s, TolZ=1e-6, axis=0)[0]

            else:
                _method = getattr(tsa, method)
                # Smoothing
                #fit = _method(y).fit(smoothing_level=s, optimized=False)
                fit = _method(y.astype(float)).fit(smoothing_level=s)
                # Re-cast to original data type
                fittedvalues = np.zeros_like(fit.fittedvalues)
                fittedvalues[0:-1] = fit.fittedvalues[1::]
                fittedvalues[-1] = y[-1]

            # Plot
            tmp_ds = img_plot_sd.copy(deep=True,
                        data=fittedvalues)
            tmp_ds.plot(ax = self.ts_p, label=method, linewidth=2)

        # Change ylimits
        max_val = img_plot_sd.data.max()
        min_val = img_plot_sd.data.min()

        data_range = max_val - min_val
        max_val = max_val + (data_range * 0.2)
        min_val = min_val - (data_range * 0.2)
        self.ts_p.set_ylim([min_val, max_val])

        # Legend
        self.ts_p.legend(loc='best', fontsize='small',
                         fancybox=True, framealpha=0.5)

        # Grid
        self.ts_p.grid(axis='both', alpha=.3)

        # Redraw plot
        plt.draw()

    @pyqtSlot(int)
    def __on_time_steps_change(self, index):
        """
        Handles a change in the time step to display
        """
        if len(self.time_steps.currentText()) == 0 or \
                self.img_imshow is None:
            return None

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        self.img_ds = getattr(self.ts.data, self.data_vars.currentText())

        self.img_imshow.set_data(self.img_ds.data[index])

        # Set titles
        self.img_p.set_title(self.time_steps.currentText())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    def __populate_plots(self):
        """
        Populate plots
        """
        self.img_ds = getattr(self.ts.data, self.data_vars.currentText())
        # Create plot
        self.img_imshow = self.img_ds[0].plot.imshow(cmap='Greys_r',
                ax=self.img_p, add_colorbar=False)

        # Turn off axis
        self.img_p.axis('off')
        self.img_p.set_aspect('equal')
        self.fig.canvas.draw_idle()

        # Connect the canvas with the event
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.on_click)

        # Plot the centroid
        _layers, _rows, _cols = self.img_ds.shape

        plot_sd = self.img_ds[:, int(_cols / 2), int(_rows / 2)]
        plot_sd.plot(ax = self.ts_p, color='black',
                linestyle = '-', linewidth=1, label='Original data')

        plt.margins(tight=True)
        plt.tight_layout()

        # Legend
        self.ts_p.legend(loc='best', fontsize='small',
                         fancybox=True, framealpha=0.5)

        # Grid
        self.ts_p.grid(axis='both', alpha=.3)

    def __create_plot_objects(self):
        """
        Create plot objects
        """
        self.fig = plt.figure(figsize=(13.5, 4.0))

        # Image plot
        self.img_p = plt.subplot2grid((1, 4), (0, 0), colspan=1)
        # Time series plot
        self.ts_p = plt.subplot2grid((1, 4), (0, 1), colspan=3)

    def _plot(self, cmap='viridis', dpi=72):
        """
        From the TATSSI Time Series Analysis object plots:
            - A single time step of the selected variable
            - Per-pixel time series with user selected smoothing
        """
        # Set plot variables
        self.__set_variables()

        # Set plot on the plot widget
        self.plotWidget = FigureCanvas(self.fig)
        # Set focus
        self.plotWidget.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.plotWidget.setFocus()
        # Connect the canvas with the event
        self.plotWidget.mpl_connect('button_press_event',
                self.on_click)

        lay = QtWidgets.QVBoxLayout(self.content_plot)
        lay.setContentsMargins(0, 100, 0, 0)
        lay.addWidget(self.plotWidget)
        # Add toolbar
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                NavigationToolbar(self.plotWidget, self))

#if __name__ == "__main__":

#    fname = '/home/glopez/Projects/TATSSI/data/MOD13A2.006/1_km_16_days_EVI/interpolated/MOD13A2.006._1_km_16_days_EVI.linear.tif'

#    app = QtWidgets.QApplication([])
#    window = TimeSeriesSmoothingUI(fname=fname)
#    app.exec_()

