
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
        as NavigationToolbar

from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import pyqtSlot

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
        # Get widgets
        self.data_vars = self.content_plot.findChild(QtWidgets.QComboBox,
                'data_vars')
        self.data_vars.currentIndexChanged.connect(
                self.__on_data_vars_change)

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

         # Create plot objects
        self.__create_plot_objects()
        # Create plot
        self.__populate_plots()

    @pyqtSlot(int)
    def __on_data_vars_change(self, change):
        """
        Handles a change in the data variable to display
        """
        from IPython import embed ; ipshell = embed()

        if change['type'] == 'change' and change['name'] == 'value':
            self.left_ds = getattr(self.ts.data, change['new'])
            if self.mask is None:
                self.right_ds = self.left_ds.copy(deep=True)
            else:
                self.right_ds = self.left_ds * self.mask

            self.left_imshow.set_data(self.left_ds.data[0])
            self.right_imshow.set_data(self.right_ds.data[0])

    def __fill_data_variables(self):
        """
        Fill the data variables dropdown list
        """
        data_vars = []
        for data_var in self.ts.data.data_vars:
            data_vars.append(data_var)

        #self.data_vars = Dropdown(
        #    options=data_vars,
        #    value=data_vars[0],
        #    description='Data variables:',
        #    disabled=False,
        #    style = {'description_width': 'initial'},
        #    layout={'width': '400px'},
        #)

        #self.data_vars.observe(self.on_data_vars_change)

        return data_vars

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

    def __populate_plots(self):
        """
        Populate plots
        """
        # Left plot
        self.left_ds = getattr(self.ts.data, self.data_vars.currentText())
        self.left_imshow = self.left_ds[0].plot.imshow(cmap='Greys_r',
                ax=self.left_p, add_colorbar=False)

        # Turn off axis
        self.left_p.axis('off')
        self.left_p.set_aspect('equal')
        self.fig.canvas.draw_idle()

        # Connect the canvas with the event
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.on_click)

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

        # Right plot
        self.right_imshow = self.right_ds[0].plot.imshow(cmap='Greys_r',
                ax=self.right_p, add_colorbar=False)

        # Turn off axis
        self.right_p.axis('off')
        self.right_p.set_aspect('equal')

        plt.margins(tight=True)
        plt.tight_layout()

        # Legend
        self.ts_p.legend(loc='best', fontsize='small',
                         fancybox=True, framealpha=0.5)

    def __create_plot_objects(self):
        """
        Create plot objects
        """
        self.fig = plt.figure(figsize=(8.0, 7.0))

        # Left plot
        self.left_p = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        # Right plot
        self.right_p = plt.subplot2grid((2, 2), (0, 1), colspan=1,
                                   sharex=self.left_p, sharey=self.left_p)
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
        lay = QtWidgets.QVBoxLayout(self.content_plot)
        lay.setContentsMargins(0, 50, 0, 0)
        lay.addWidget(self.plotWidget)
        # Add toolbar
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                NavigationToolbar(self.plotWidget, self))

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

        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9,5),
                sharex=True, sharey=True, tight_layout=True, dpi=dpi)

        qa_analytics.pct_data_available.plot.imshow(
                ax=ax, cmap=cmap,
                cbar_kwargs={'orientation':'horizontal',
                             'pad' : 0.005},
        )

        ax.set_frame_on(False)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.title.set_text('% of data available')
        ax.margins(tight=True)

        qa_analytics.max_gap_length.plot.imshow(
                ax=bx, cmap=cmap,
                cbar_kwargs={'orientation':'horizontal',
                             'pad' : 0.005},
        )

        bx.set_frame_on(False)
        bx.axis('off')
        bx.set_aspect('equal')
        bx.title.set_text('Max gap-length')
        bx.margins(tight=True)

        # Set plot on the plot widget
        self.plotWidget = FigureCanvas(fig)
        lay = QtWidgets.QVBoxLayout(self.content_plot)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)
        # Add toolbar
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                NavigationToolbar(self.plotWidget, self))
