
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.notebooks.helpers.time_series_interpolation import \
        TimeSeriesInterpolation

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
        as NavigationToolbar

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import osr

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

class PlotInterpolation(QtWidgets.QMainWindow):
    """
    Plot time series interpolation tool
    """
    def __init__(self, parent=None):
        super(PlotInterpolation, self).__init__(parent)

    def __set_variables(self, qa_analytics):
        """
        Set variables from TATSSI QA analytics
        """
        # Set qa_analytics
        self.qa_analytics = qa_analytics

        # imshow plots
        self.left_imshow = None
        self.right_imshow = None
        self.projection = None

        # Set widgets connections with methods
        self.data_vars.currentIndexChanged.connect(
                self.__on_data_vars_change)

        self.time_steps.currentIndexChanged.connect(
                self.__on_time_steps_change)

        self.pb_Interpolate.clicked.connect(
                self.on_pbInterpolate_click)

        self.interpolation_methods.clicked.connect(
                self.on_interpolation_methods_click)

        self.progressBar.setEnabled(False)
        self.pb_Interpolate.setEnabled(False)

        # Change combobox stylesheet and add scrollbar
        self.time_steps.setStyleSheet("combobox-popup: 0")
        self.time_steps.view().setVerticalScrollBarPolicy(
                Qt.ScrollBarAsNeeded)

        # Time series object
        self.ts = qa_analytics.ts
        # Source dir
        self.source_dir = qa_analytics.source_dir
        # Product and version
        self.product = qa_analytics.product
        self.version = qa_analytics.version

        # Mask
        self.mask = qa_analytics.mask

        # Data variables
        self.data_vars.addItems(self.__fill_data_variables())
        # Time steps
        self.time_steps.addItems(self.__fill_time_steps())

         # Create plot objects
        self.__create_plot_objects()

        # Populate plots
        self.__populate_plots()

        # Set interpolation methods
        self.__fill_interpolation_methods()

    def __fill_interpolation_methods(self):
        """
        Fill interpolation methods
        """
        interpolation_methods = ['linear', 'nearest', 'slinear',
                                 'quadratic', 'cubic', 'krog',
                                 'pchip', 'spline', 'akima']

        self.interpolation_methods.addItems(interpolation_methods)

    def on_pbInterpolate_click(self):
        """
        Performs interpolation for using a specific user selected
        method taking into account the TATSSI QA analytics mask
        """
        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        # Variables for interpolation
        self.qa_analytics.selected_data_var = \
                self.data_vars.currentText()
        self.qa_analytics.selected_interpolation_method = \
                self.interpolation_methods.selectedItems()[0].text()

        # TATSSI interpolation
        tsi = TimeSeriesInterpolation(self.qa_analytics, isNotebook=False)
        # Enable progress bar
        self.progressBar.setEnabled(True)
        self.progressBar.setValue(0)
        tsi.interpolate(progressBar=self.progressBar)

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

        # Disable progress bar
        self.progressBar.setValue(0)
        self.progressBar.setEnabled(False)

    def on_interpolation_methods_click(self):
        """
        Enable the pbInterpolate push button when there is an
        interpolation method selected
        """
        if len(self.interpolation_methods.selectedItems()) > 0:
            self.pb_Interpolate.setEnabled(True)
        else:
            self.pb_Interpolate.setEnabled(False)

    @pyqtSlot(int)
    def __on_time_steps_change(self, index):
        """
        Handles a change in the time step to display
        """
        if len(self.time_steps.currentText()) == 0 or \
                self.left_imshow is None or \
                self.right_imshow is None:
            return None

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        self.left_ds = getattr(self.ts.data, self.data_vars.currentText())
        if self.mask is None:
            self.right_ds = self.left_ds.copy(deep=True)
        else:
            self.right_ds = self.left_ds * self.mask

        self.left_imshow.set_data(self.left_ds.data[index])
        self.right_imshow.set_data(self.right_ds.data[index])

        # Set titles
        self.left_p.set_title(self.time_steps.currentText())
        self.right_p.set_title(self.time_steps.currentText())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    @pyqtSlot(int)
    def __on_data_vars_change(self, index):
        """
        Handles a change in the data variable to display
        """
        if len(self.data_vars.currentText()) == 0 or \
                self.left_imshow is None or \
                self.right_imshow is None:
            return None

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        self.left_ds = getattr(self.ts.data, self.data_vars.currentText())
        if self.mask is None:
            self.right_ds = self.left_ds.copy(deep=True)
        else:
            self.right_ds = self.left_ds * self.mask

        self.left_imshow.set_data(self.left_ds.data[0])
        self.right_imshow.set_data(self.right_ds.data[0])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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
        if len(self.left_p.lines) > 0:
            del self.left_p.lines[0]
            del self.right_p.lines[0]

        # Draw a point as a reference
        # Draw a point as a reference
        self.left_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=7, alpha=0.7)
        self.right_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=7, alpha=0.7)

        # Non-masked data
        left_plot_sd = self.left_ds.sel(longitude=event.xdata,
                                        latitude=event.ydata,
                                        method='nearest')
        if left_plot_sd.chunks is not None:
            left_plot_sd = left_plot_sd.compute()

        # Masked data
        right_plot_sd = self.right_ds.sel(longitude=event.xdata,
                                          latitude=event.ydata,
                                          method='nearest')
        if right_plot_sd.chunks is not None:
            right_plot_sd = right_plot_sd.compute()

        # Plots
        left_plot_sd.plot(ax=self.ts_p, color='black',
                linestyle = '-', linewidth=1, label='Original data')

        # Interpolate data
        right_plot_sd_masked = right_plot_sd.where(right_plot_sd != 0)
        right_plot_sd_masked.plot(ax = self.ts_p, color='blue',
                marker='o', linestyle='None', alpha=0.7, markersize=4,
                label='Kept by user QA selection')

        # For every interpol method selected by the user
        for method in self.interpolation_methods.selectedItems():
            _method=method.text()
            tmp_ds = right_plot_sd_masked.interpolate_na(dim='time',
                    method=_method)

            # Plot
            tmp_ds.plot(ax = self.ts_p, label=_method, linewidth=2)

        # Change ylimits
        max_val = left_plot_sd.data.max()
        min_val = left_plot_sd.data.min()

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

    def __populate_plots(self):
        """
        Populate plots
        """
        # Left plot
        self.left_ds = getattr(self.ts.data, self.data_vars.currentText())
        self.left_imshow = self.left_ds[0].plot.imshow(cmap='Greys_r',
                ax=self.left_p, add_colorbar=False,
                transform=self.projection)

        # Turn off axis
        self.left_p.axis('off')
        self.left_p.set_aspect('equal')
        self.fig.canvas.draw_idle()

        # Plot the centroid
        _layers, _rows, _cols = self.left_ds.shape

        plot_sd = self.left_ds[:, int(_cols / 2), int(_rows / 2)]
        plot_sd.plot(ax = self.ts_p, color='black',
                linestyle = '--', linewidth=1, label='Original data')

        # Right panel
        if self.mask is None:
            self.right_ds = self.left_ds.copy(deep=True)
        else:
            self.right_ds = self.left_ds * self.mask
            self.right_ds.attrs = self.left_ds.attrs

        # Right plot
        self.right_imshow = self.right_ds[0].plot.imshow(cmap='Greys_r',
                ax=self.right_p, add_colorbar=False,
                transform=self.projection)

        # Turn off axis
        self.right_p.axis('off')
        self.right_p.set_aspect('equal')

        #plt.margins(tight=True)
        #plt.tight_layout()

        # Legend
        self.ts_p.legend(loc='best', fontsize='small',
                         fancybox=True, framealpha=0.5)

    def __create_plot_objects(self):
        """
        Create plot objects
        """
        # Get projection from first data variable
        for key in self.qa_analytics.ts.data.data_vars:
            proj4_string = getattr(self.qa_analytics.ts.data, key).crs
            break

        # If projection is Sinusoidal
        srs = get_projection(proj4_string)
        if srs.GetAttrValue('PROJECTION') == 'Sinusoidal':
            globe=ccrs.Globe(ellipse=None,
                semimajor_axis=6371007.181,
                semiminor_axis=6371007.181)

            self.projection = ccrs.Sinusoidal(globe=globe)

        # Figure
        self.fig = plt.figure(figsize=(8.0, 7.0))

        # Left plot
        self.left_p = plt.subplot2grid((2, 2), (0, 0), colspan=1,
                projection=self.projection)

        # Right plot
        self.right_p = plt.subplot2grid((2, 2), (0, 1), colspan=1,
                sharex=self.left_p, sharey=self.left_p,
                projection=self.projection)

        if self.projection is not None:
            for _axis in [self.left_p, self.right_p]:
                _axis.coastlines(resolution='10m', color='white')
                _axis.add_feature(cfeature.BORDERS, edgecolor='white')
                _axis.gridlines()

        # Time series plot
        self.ts_p = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    def _plot(self, qa_analytics, cmap='viridis', dpi=72):
        """
        From the TATSSI QA Analytics object plots:
          - Percentage of data available
          - Maximum gap length
        """
        # Load UI
        uic.loadUi('plot_interpolation.ui', self)

        # Set plot variables
        self.__set_variables(qa_analytics)

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
        font = QFont()
        font.setPointSize(12)

        toolbar = NavigationToolbar(self.plotWidget, self)
        toolbar.setFont(font)

        self.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

        # Needed in order to use a tight layout with Cartopy axes
        self.fig.canvas.draw()
        plt.tight_layout()

class PlotStatistics(QtWidgets.QMainWindow):
    """
    Plot the histogram of the maximum gap-length and the percentage of
    data available when applying a QA-user selection to a TATSSI time
    series
    """
    def __init__(self, parent=None):
        super(PlotStatistics, self).__init__(parent)

    def _plot(self, qa_analytics, dpi=72):
        """
        From the TATSSI QA Analytics object plots histograms of:
          - Percentage of data available
          - Maximum gap length
        """
        uic.loadUi('plot.ui', self)

        fig, (ax, bx) = plt.subplots(1, 2, figsize=(16,11),
                sharex=True, sharey=True, tight_layout=True, dpi=dpi)

        # Plot histograms
        qa_analytics.pct_data_available.plot.hist(ax=ax)
        qa_analytics.max_gap_length.plot.hist(ax=bx)

        ax.title.set_text('% of data available histogram')
        bx.title.set_text('Max gap-length histogram')

        # Set plot on the plot widget
        self.plotWidget = FigureCanvas(fig)
        lay = QtWidgets.QVBoxLayout(self.content_plot)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)

        # Add toolbar
        font = QFont()
        font.setPointSize(12)

        toolbar = NavigationToolbar(self.plotWidget, self)
        toolbar.setFont(font)

        self.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

        plt.tight_layout()

class PlotMaxGapLength(QtWidgets.QMainWindow):
    """
    Plot the maximum gap-length and the percentage of data available
    when applying a QA-user selection to a TATSSI time series
    """
    def __init__(self, parent=None):
        super(PlotMaxGapLength, self).__init__(parent)

    def _plot(self, qa_analytics, cmap='viridis', dpi=72):
        """
        From the TATSSI QA Analytics object plots:
          - Percentage of data available
          - Maximum gap length
        """
        uic.loadUi('plot.ui', self)

        # Get projection from first data variable
        for key in qa_analytics.ts.data.data_vars:
            proj4_string = getattr(qa_analytics.ts.data, key).crs
            break

        # If projection is Sinusoidal
        srs = get_projection(proj4_string)
        if srs.GetAttrValue('PROJECTION') == 'Sinusoidal':
            globe=ccrs.Globe(ellipse=None,
                semimajor_axis=6371007.181,
                semiminor_axis=6371007.181)

            proj = ccrs.Sinusoidal(globe=globe)
        else:
            proj = None

        fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 9.6),
                #sharex=True, sharey=True, tight_layout=True, dpi=dpi,
                sharex=True, sharey=True, dpi=dpi,
                subplot_kw=dict(projection=proj))

        if proj is not None:
            for _axis in [ax, bx]:
                _axis.coastlines(resolution='10m', color='white')
                _axis.add_feature(cfeature.BORDERS, edgecolor='white')
                _axis.gridlines()

        qa_analytics.pct_data_available.plot.imshow(
                ax=ax, cmap=cmap,
                cbar_kwargs={'orientation':'horizontal',
                             'pad' : 0.005},
                transform=proj
        )

        ax.set_frame_on(False)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.title.set_text('% of data available')

        qa_analytics.max_gap_length.plot.imshow(
                ax=bx, cmap=cmap,
                cbar_kwargs={'orientation':'horizontal',
                             'pad' : 0.005},
                transform=proj
        )

        bx.set_frame_on(False)
        bx.axis('off')
        bx.set_aspect('equal')
        bx.title.set_text('Max gap-length')

        # Set plot on the plot widget
        self.plotWidget = FigureCanvas(fig)
        lay = QtWidgets.QVBoxLayout(self.content_plot)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)

        # Add toolbar
        font = QFont()
        font.setPointSize(12)

        toolbar = NavigationToolbar(self.plotWidget, self)
        toolbar.setFont(font)

        self.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

        # Needed in order to use a tight layout with Cartopy axes
        fig.canvas.draw()
        plt.tight_layout()


