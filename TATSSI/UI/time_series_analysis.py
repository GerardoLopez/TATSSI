
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.time_series.smoothn import smoothn
from TATSSI.time_series.analysis import Analysis
from TATSSI.time_series.mk_test import mk_test
from TATSSI.UI.plots_time_series_analysis import PlotAnomalies
from TATSSI.UI.helpers.utils import *

#from TATSSI.notebooks.helpers.time_series_analysis import \
#        TimeSeriesAnalysis

import ogr
import numpy as np
from rasterio import logging as rio_logging
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks

import seaborn as sbn

from dask.diagnostics import ProgressBar

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
        as NavigationToolbar

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader as cReader
from cartopy.feature import ShapelyFeature
import osr

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont

# Experimental
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from rpy2.robjects import numpy2ri

def get_projection(proj4_string):
    """
    Get spatial reference system from PROJ4 string
    """
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4_string)

    return srs

class TimeSeriesAnalysisUI(QtWidgets.QMainWindow):
    def __init__(self, fname, parent=None):
        super(TimeSeriesAnalysisUI, self).__init__(parent)
        uic.loadUi('time_series_analysis.ui', self)
        self.parent = parent

        # List of dialogs
        self.dialogs = list()

        # Set input file name
        self.fname = fname

        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

        # Change point R library
        self.cpt = importr('changepoint')

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
        self.left_imshow = None
        self.right_imshow = None
        self.projection = None

        # Connect time steps with corresponsing method
        self.time_steps_left.currentIndexChanged.connect(
                self.__on_time_steps_left_change)

        self.time_steps_right.currentIndexChanged.connect(
                self.__on_time_steps_right_change)

        # Change time_steps combo boxes stylesheet and add scrollbar
        self.time_steps_left.setStyleSheet("combobox-popup: 0")
        self.time_steps_left.view().setVerticalScrollBarPolicy(
                Qt.ScrollBarAsNeeded)

        self.time_steps_right.setStyleSheet("combobox-popup: 0")
        self.time_steps_right.view().setVerticalScrollBarPolicy(
                Qt.ScrollBarAsNeeded)

        # Climatology button
        self.pbAnomalies.clicked.connect(
                self.on_pbAnomalies_click)
        # Overlay button
        self.pbOverlay.clicked.connect(
                self.on_pbOverlay_click)
        # Save products button
        self.pbSaveProducts.clicked.connect(
                self.on_pbSaveProducts_click)

        # Data variables
        self.data_vars.addItems(self.__fill_data_variables())

        # Years
        self.__fill_year()
        self.years.currentIndexChanged.connect(
                self.__on_years_change)

        # Time steps
        self.time_steps_left.addItems(self.__fill_time_steps())
        self.time_steps_right.addItems(self.__fill_time_steps())

        # Decomposition model
        self.model.addItems(self.__fill_model())

        # Create plot objects
        self.__create_plot_objects()

        # Populate plots
        self.__populate_plots()

        # Climatology year
        self.climatology_year = None
        self.anomalies = None

    @pyqtSlot(int)
    def __on_time_steps_change(self, index):
        """
        Handles a change in the time step to display
        """
        if len(self.time_steps_left.currentText()) == 0 or \
                len(self.time_steps_right.currentText()) == 0 or \
                self.left_imshow is None or \
                self.right_imshow is None:
            return None

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        self.left_imshow.set_data(self.self.single_year_ds.data[index])
        self.right_imshow.set_data(self.self.single_year_ds.data[index])

        # Set titles
        self.left_p.set_title(self.time_steps_left.currentText())
        self.right_p.set_title(self.time_steps_right.currentText())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    def on_pbSaveProducts_click(self):
        """
        Save to COGs all the time series analysis products
        """
        n_quartiles = 3
        quartiles = [0.25, 0.5, 0.75]
        var = '_1_km_16_days_EVI'

        # Group by day of year to compute quartiles
        grouped_by_doy = self.ts.data[var].time.groupby("time.dayofyear")
        # Get the number of time steps
        n_time_steps = len(grouped_by_doy.groups.keys())
        # rows and cols
        rows, cols = self.ts.data[var].shape[1:]
        # Create output array
        # Layers
        # 0 - Q1
        # 1 - Median
        # 2 - Q3
        # 3 - Q1 - 1.5 * IQR
        # 4 - Q3 + 1.5 * IQR
        # 5 - Negative outliers
        # 6 - Positive outliers
        q_data = np.zeros((n_quartiles+4, n_time_steps, rows, cols))

        for i, (doy, _times) in enumerate(grouped_by_doy):
            # Get time series for DoY
            ts_doy = self.ts.data[var].sel(time=_times.data)
            # Get quartiles for DoY
            q_data[0:3,i] = np.quantile(ts_doy, q=quartiles, axis=0)

            # Get IQR
            iqr = q_data[2,i] - q_data[0,i]
            # Get boundaries
            q_data[3,i] = q_data[0,i] - (1.5 * iqr)
            q_data[4,i] = q_data[2,i] + (1.5 * iqr)
            # Get outliers

            from IPython import embed ; ipshell = embed()
            np.where(ts_doy < q_data[3,i])[0] 


        from IPython import embed ; ipshell = embed()

    def on_pbOverlay_click(self):
        """
        EXPERIMENTAL
        Overlay a specific geometry on maps
        """
        fname = open_file_dialog(dialog_type = 'open_specific',
                data_format = 'Shapefile',
                extension = 'shp')

        if os.path.exists(fname) is False:
            pass

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

            for _axis in [self.left_p, self.right_p]:
                _axis.add_feature(shape_feature,
                        edgecolor='gray')
        except:
            return None

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_pbAnomalies_click(self):
        """
        Computes the climatolgy and anomalies for the selected year
        and shows the correspinding plot
        """
        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        # The get_climatology method will create two datasets:
        #   ts.climatology_mean
        #   ts.climatology_std
        if self.ts.climatology_mean is None and \
            self.ts.climatology_std is None and \
            (self.climatology_year is None or \
             self.climatology_year != self.years.currentText()):

            # Compute climatology
            self.ts.climatology()

            with ProgressBar():
                self.ts.climatology_mean = self.ts.climatology_mean.compute()
                self.ts.climatology_std = self.ts.climatology_std.compute()

            self.climatology_year = self.years.currentText()

            if self.ts.climatology_mean.shape[0] != self.single_year_ds.shape[0]:
                # Standard cursor
                QtWidgets.QApplication.restoreOverrideCursor()

                message_text = (f'Year {self.years.currentText()} does not '
                                f'have same number of observations as the '
                                f'climatology. Anomalies cannot be computed.')
                self.message_box(message_text)
                return None

            # Anomalies
            anomalies = (self.single_year_ds - self.ts.climatology_mean.data) \
                            / self.ts.climatology_std.data

            with ProgressBar():
                self.anomalies = anomalies.compute()

        self.__plotAnomalies()

        # Output file name
        #smoothing_method = self.smoothing_methods.selectedItems()[0].text()
        #_fname, _ext = os.path.splitext(self.fname)
        #output_fname = f'{_fname}.{smoothing_method}.tif'

        #smoother = Smoothing(fname=self.fname, output_fname=output_fname,
        #        smoothing_methods=[smoothing_method])

        #smoother.smooth()

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    @pyqtSlot()
    def __plotAnomalies(self):
        dialog = PlotAnomalies(self)
        self.dialogs.append(dialog)
        dialog._plot(self.anomalies, self.years.currentText())
        dialog.show()

    def __fill_year(self):
        """
        Fill years list based on years in the dataset
        """
        # Get unique years from data
        times = getattr(self.ts.data, self.data_vars.currentText()).time
        times = np.unique(times.dt.year.data).tolist()

        # We need a list of strings to be added to the widget
        times = list(map(str, times))
        self.years.addItems(times)

        current_year = times[0]

        time_slice = slice(f"{current_year}-01-01",
                           f"{current_year}-12-31",)

        self.single_year_ds = getattr(self.ts.data,
                self.data_vars.currentText()).sel(time=time_slice)

    def __fill_model(self):
        """
        Fill time series decompostion model
        """
        _models = ['additive', 'multiplicative']

        return _models

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
        time_steps = np.datetime_as_string(
                self.single_year_ds.time.data, 'm').tolist()

        return time_steps

    def on_click(self, event):
        """
        Event handler
        """
        # Event does not apply for time series plot
        if event.inaxes not in [self.left_p, self.right_p]:
            return

        # Clear subplots
        self.observed.clear()
        self.trend.clear()
        self.seasonal.clear()
        self.resid.clear()
        self.climatology.clear()

        # Delete last reference point
        if len(self.left_p.lines) > 0:
            del self.left_p.lines[0]
            del self.right_p.lines[0]

        # Draw a point as a reference
        self.left_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=7, alpha=0.7)
        self.right_p.plot(event.xdata, event.ydata,
                marker='o', color='red', markersize=7, alpha=0.7)

        # Non-masked data
        left_plot_sd = self.left_ds.sel(longitude=event.xdata,
                                        latitude=event.ydata,
                                        method='nearest')
        # Sinlge year dataset
        single_year_ds = self.single_year_ds.sel(longitude=event.xdata,
                                                 latitude=event.ydata,
                                                 method='nearest')

        if left_plot_sd.chunks is not None:
            left_plot_sd = left_plot_sd.compute()
            single_year_ds = single_year_ds.compute()

        # Seasonal decompose
        ts_df = left_plot_sd.to_dataframe()
        self.seasonal_decompose = seasonal_decompose(
                ts_df[self.data_vars.currentText()],
                model=self.model.currentText(),
                freq=self.single_year_ds.shape[0],
                extrapolate_trend='freq')

        # Plot seasonal decompose
        self.observed.plot(self.seasonal_decompose.observed.index,
                self.seasonal_decompose.observed.values,
                label='Observed')

        peaks, _ = find_peaks(ts_df[self.data_vars.currentText()])
                #height=2000)
        self.observed.plot(self.seasonal_decompose.observed.index[peaks],
                ts_df[self.data_vars.currentText()][peaks],
                label=f'Peaks [{peaks.shape[0]}]',
                marker='x', color='C1', alpha=0.5)

        # MK test
        #_mk_test = mk_test(self.seasonal_decompose.trend)
        self.trend.plot(self.seasonal_decompose.trend.index,
                self.seasonal_decompose.trend.values,
                label=f'Trend')
                #label=f'Trend {_mk_test}')

        # Set the same y limits from observed data
        self.trend.set_ylim(self.observed.get_ylim())

        self.seasonal.plot(self.seasonal_decompose.seasonal.index,
                self.seasonal_decompose.seasonal.values,
                label='Seasonality')
        self.resid.plot(self.seasonal_decompose.resid.index,
                self.seasonal_decompose.resid.values,
                label='Residuals')

        # Climatology
        sbn.boxplot(ts_df.index.dayofyear,
                ts_df[self.data_vars.currentText()], ax=self.climatology)
        # Plot year to analyse
        single_year_df = single_year_ds.to_dataframe()
        sbn.stripplot(x=single_year_df.index.dayofyear,
                y=single_year_df[self.data_vars.currentText()],
                color='red', marker='o', size=7, alpha=0.7,
                ax=self.climatology)

        self.climatology.tick_params(axis='x', rotation=70)

        # Change point
        r_vector = FloatVector(self.seasonal_decompose.trend.values)
        #changepoint_r = self.cpt.cpt_mean(r_vector)
        #changepoints_r = self.cpt.cpt_var(r_vector, method='PELT',
        #        penalty='Manual', pen_value='2*log(n)')
        changepoints_r = self.cpt.cpt_meanvar(r_vector,
                test_stat='Normal', method='BinSeg', penalty="SIC")
        changepoints = numpy2ri.rpy2py(self.cpt.cpts(changepoints_r))

        if changepoints.shape[0] > 0:
            # Plot vertical line where the changepoint was found
            for i, i_cpt in enumerate(changepoints):
                i_cpt = int(i_cpt) + 1
                cpt_index = self.seasonal_decompose.trend.index[i_cpt]
                if i == 0 :
                    self.trend.axvline(cpt_index, color='black',
                            lw='1.0', label='Change point')
                else:
                    self.trend.axvline(cpt_index, color='black', lw='1.0')

        # Legend
        self.observed.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.observed.set_title('Time series decomposition')
        self.trend.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.seasonal.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)
        self.resid.legend(loc='best', fontsize='small',
                fancybox=True, framealpha=0.5)

        #self.climatology.legend([self.years.value], loc='best',
        self.climatology.legend(loc='best',
                fontsize='small', fancybox=True, framealpha=0.5)

        self.climatology.set_title('Climatology')

        # Grid
        self.observed.grid(axis='both', alpha=.3)
        self.trend.grid(axis='both', alpha=.3)
        self.seasonal.grid(axis='both', alpha=.3)
        self.resid.grid(axis='both', alpha=.3)
        self.climatology.grid(axis='both', alpha=.3)

        # Redraw plot
        plt.draw()

    @pyqtSlot(int)
    def __on_years_change(self,  index):
        """
        Handles a change in the years to display
        """
        if len(self.years.currentText()) == 0 or \
                self.left_imshow is None or \
                self.right_imshow is None:
            return None

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        current_year = self.years.currentText()

        time_slice = slice(f"{current_year}-01-01",
                           f"{current_year}-12-31",)
        self.single_year_ds = getattr(self.ts.data,
                self.data_vars.currentText()).sel(time=time_slice)

        # Update time steps
        self.time_steps_left.clear() ; self.time_steps_right.clear()
        self.time_steps_left.addItems(self.__fill_time_steps())
        self.time_steps_right.addItems(self.__fill_time_steps())

        # Update images with current year
        self.__update_imshow()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    @pyqtSlot(int)
    def __on_time_steps_left_change(self, index):
        """
        Handles a change in the time step to display
        """
        if len(self.time_steps_left.currentText()) == 0 or \
                self.left_imshow is None:
            return None

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        self.left_imshow.set_data(self.single_year_ds.data[index])

        # Set titles
        self.left_p.set_title(self.time_steps_left.currentText())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    @pyqtSlot(int)
    def __on_time_steps_right_change(self, index):
        """
        Handles a change in the time step to display
        """
        if len(self.time_steps_right.currentText()) == 0 or \
                self.right_imshow is None:
            return None

        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        self.right_imshow.set_data(self.single_year_ds.data[index])

        # Set titles
        self.right_p.set_title(self.time_steps_right.currentText())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    def __update_imshow(self):
        """
        Update images shown as imshow plots
        """
        self.left_imshow = self.single_year_ds[0].plot.imshow(
                cmap=self.cmap, ax=self.left_p, add_colorbar=False,
                transform=self.projection)

        self.right_imshow = self.single_year_ds[0].plot.imshow(
                cmap=self.cmap, ax=self.right_p, add_colorbar=False,
                transform=self.projection)

        self.left_p.set_aspect('equal')
        self.right_p.set_aspect('equal')

    def __populate_plots(self):
        """
        Populate plots
        """
        # Create plot
        self.left_ds = getattr(self.ts.data, self.data_vars.currentText())
        self.right_ds = getattr(self.ts.data, self.data_vars.currentText())

        # Show images
        self.__update_imshow()

        # Turn off axis
        self.left_p.axis('off')
        self.right_p.axis('off')
        self.fig.canvas.draw_idle()

        # Connect the canvas with the event
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.on_click)

        # Plot the centroid
        _layers, _rows, _cols = self.left_ds.shape

        # Seasonal decompose
        left_plot_sd = self.left_ds[:, int(_cols / 2), int(_rows / 2)]
        ts_df = left_plot_sd.to_dataframe()
        self.seasonal_decompose = seasonal_decompose(
                ts_df[self.data_vars.currentText()],
                model=self.model.currentText(),
                freq=self.single_year_ds.shape[0],
                extrapolate_trend='freq')

        # Plot seasonal decompose
        self.seasonal_decompose.observed.plot(ax=self.observed)
        self.seasonal_decompose.trend.plot(ax=self.trend)
        self.seasonal_decompose.seasonal.plot(ax=self.seasonal)
        self.seasonal_decompose.resid.plot(ax=self.resid)

        # Climatology
        sbn.boxplot(ts_df.index.dayofyear,
                ts_df[self.data_vars.currentText()],
                ax=self.climatology)
        self.climatology.tick_params(axis='x', rotation=70)

        plt.margins(tight=True)
        plt.tight_layout()

    def __create_plot_objects(self):
        """
        Create plot objects
        """
        years_fmt = mdates.DateFormatter('%Y')

        # Get projection from first data variable
        for key in self.ts.data.data_vars:
            proj4_string = getattr(self.ts.data, key).crs
            break

        # If projection is Sinusoidal
        srs = get_projection(proj4_string)
        if srs.GetAttrValue('PROJECTION') == 'Sinusoidal':
            globe=ccrs.Globe(ellipse=None,
                semimajor_axis=6371007.181,
                semiminor_axis=6371007.181)

            self.projection = ccrs.Sinusoidal(globe=globe)

        # Figure
        self.fig = plt.figure(figsize=(11.0, 6.0))

        # subplot2grid((rows,cols), (row,col)
        # Left, right and climatology plots
        self.left_p = plt.subplot2grid((4, 4), (0, 0), rowspan=2,
                projection=self.projection)
        self.right_p = plt.subplot2grid((4, 4), (0, 1), rowspan=2,
                sharex=self.left_p, sharey=self.left_p,
                projection=self.projection)

        if self.projection is not None:
            for _axis in [self.left_p, self.right_p]:
                _axis.coastlines(resolution='10m', color='white')
                _axis.add_feature(cfeature.BORDERS, edgecolor='white')
                _axis.gridlines()

        self.climatology = plt.subplot2grid((4, 4), (2, 0),
                rowspan=2, colspan=2)

        # Time series plot
        self.observed = plt.subplot2grid((4, 4), (0, 2), colspan=2)
        self.observed.xaxis.set_major_formatter(years_fmt)

        self.trend = plt.subplot2grid((4, 4), (1, 2), colspan=2,
                sharex=self.observed)
        self.seasonal = plt.subplot2grid((4, 4), (2, 2), colspan=2,
                sharex=self.observed)
        self.resid = plt.subplot2grid((4, 4), (3, 2), colspan=2,
                sharex=self.observed)

    def _plot(self, cmap='viridis', dpi=72):
        """
        From the TATSSI Time Series Analysis object plots:
            - A single time step of the selected variable
            - Per-pixel time series with user selected smoothing
        """
        # Default colormap
        self.cmap = cmap

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
        lay.setContentsMargins(0, 70, 0, 0)
        lay.addWidget(self.plotWidget)

        # Add toolbar
        font = QFont()
        font.setPointSize(12)

        toolbar = NavigationToolbar(self.plotWidget, self)
        toolbar.setFont(font)

        self.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

    @staticmethod
    def message_box(message_text):
        dialog = QtWidgets.QMessageBox()
        dialog.setIcon(QtWidgets.QMessageBox.Critical)
        dialog.setText(message_text)
        dialog.addButton(QtWidgets.QMessageBox.Ok)
        dialog.exec()

        return None

    @staticmethod
    def get_shapefile_spatial_reference(fname):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataset = driver.Open(fname)

        layer = dataset.GetLayer()
        spatialRef = layer.GetSpatialRef()

        return spatialRef

if __name__ == "__main__":

    fname = '/home/glopez/Projects/TATSSI/data/MOD13A2.006/1_km_16_days_EVI/interpolated/MOD13A2.006._1_km_16_days_EVI.linear.smoothn.int16.tif'

    app = QtWidgets.QApplication([])
    window = TimeSeriesAnalysisUI(fname=fname)
    app.exec_()

