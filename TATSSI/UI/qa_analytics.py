
import os
import sys

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
        as NavigationToolbar

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.input_output.utils import *
from TATSSI.notebooks.helpers.utils import *
from TATSSI.notebooks.helpers.qa_analytics import Analytics
from TATSSI.qa.EOS.catalogue import Catalogue
from TATSSI.UI.helpers.utils import *

import ogr
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot

import collections

class PlotMaxGapLength(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(PlotMaxGapLength, self).__init__(parent)

    def _plot(self, qa_analytics, cmap='viridis', dpi=72):
        """
        From the TATSSI QA Analytics obhect plots:
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

class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('qa_analytics.ui', self)

        # List of dialogs
        self.dialogs = list()

        # QA analytics - set on on_pbGetQADefinitions_click
        self.qa_analytics = None

        # User's QA selection - set on
        #  fill_analytics and change dynamicaly based on user
        #  selection on update_qa_param_desc_selection
        self.user_qa_selection = None
        
        # Connect methods with events
        self.pbPlotMaxGapLength.clicked.connect(
                self.on_pbPlotMaxGapLength_click)

        self.pbOutputDirectory.clicked.connect(
                self.on_pbOutputDirectory_click)

        self.pbGetQADefinitions.clicked.connect(
                self.on_pbGetQADefinitions_click)

        self.pbQAAnalytics.clicked.connect(
                self.on_pbQAAnalytics_click)

        self.pbSaveQAAnalytics.clicked.connect(
                self.on_pbSaveQAAnalytics_click)

        self.pbLoadQAAnalytics.clicked.connect(
                self.on_pbLoadQAAnalytics_click)

        self.cmbQAParamName.currentIndexChanged.connect(
                self.update_qa_param_def_description)

        self.cmbQADef.currentIndexChanged.connect(
                self.update_qa_definition)

        self.lwQAParamDesc.itemSelectionChanged.connect(
                self.update_qa_param_desc_selection)
        # Flag to avoid this event to be triggered when
        # the list is cleared
        self.lwQAParamDescUpdateInProgress = False

        # Set default date display format
        self.start_date.setDisplayFormat('dd-MM-yyyy')
        self.end_date.setDisplayFormat('dd-MM-yyyy')

        # Hide progress bar
        self.progressBar.hide()

        self.show()

    @pyqtSlot()
    def on_pbPlotMaxGapLength_click(self):
        dialog = PlotMaxGapLength(self)
        self.dialogs.append(dialog)
        dialog._plot(self.qa_analytics)
        dialog.show()

    @pyqtSlot()
    def on_pbSaveQAAnalytics_click(self):
        """
        Save the current user-defined QA setting into a JSON file
        """
        fname = open_file_dialog(dialog_type = 'save',
                data_format = 'JSON',
                extension = 'json')

        with open(fname, 'w') as f:
            f.write(json.dumps(self.user_qa_selection))

        #LOG.info(f"QA settings file {fname} written to disk.")

    @pyqtSlot()
    def on_pbLoadQAAnalytics_click(self):
        """
        Load user-defined QA saved settings from a JSON file
        """
        fname = open_file_dialog(dialog_type = 'open_specific',
                data_format = 'JSON',
                extension = 'json')

        if os.path.exists(fname) is False:
            pass

        # Open file
        try:
            with open(fname, 'r') as f:
                tmp_user_qa_selection = collections.OrderedDict(
                        json.loads(f.read()))
        except json.JSONDecodeError:
            message_text = (f'The file {fname} is not a valid '
                            f'TATSSI QA Analytics file')
            message_box(message_text)
            return None

        # Check that file has same keys as current QA selection
        if self.user_qa_selection.keys() != tmp_user_qa_selection.keys():
            message_text = (f'The file {fname} does not have the same QA '
                            f'parameters names as current QA definition')
            message_box(message_text)
            return None

        # Update user QA selection
        self.user_qa_selection = tmp_user_qa_selection

        # Update list widget lwQAParamDesc
        self.update_qa_param_def_description()
        # Update text browser tbrSelection
        self.tbrSelection.setText(json.dumps(self.user_qa_selection,
             indent=2))

    @pyqtSlot()
    def on_pbQAAnalytics_click(self):
        """
        Perform QA analytics based on user selection
        """
        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        # Set qa_analytics
        self.qa_analytics.user_qa_selection = self.user_qa_selection
        # Launch analytics
        self.progressBar.show()
        self.progressBar.setEnabled(True)
        self.progressBar.setValue(0)
        self.progressBar.setRange(0, len(self.user_qa_selection))

        self.qa_analytics._analytics(self.progressBar)

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    @pyqtSlot()
    def set_default_qa_param_desc_selection(self):
        """
        Set default QA param description selection. The first item
        of each description will be selected.
        """
        qa_def = self.cmbQADef.currentText()
        index = self.cmbQADef.findText(qa_def, QtCore.Qt.MatchFixedString)
        qa_params = self.qa_analytics.qa_defs[index].Name.unique().tolist()

        # Set user_qa_selection
        self.user_qa_selection = collections.OrderedDict(
                (element, '') for element in qa_params)

    @pyqtSlot()
    def update_qa_param_desc_selection(self):
        """
        Update the QA parameters description selected (stored in the
        user_qa_selection ordered dic) based on user's selection
        """
        if self.lwQAParamDescUpdateInProgress == False:
            return

        # Get current QA parameter name
        current_qa_param = self.cmbQAParamName.currentText()

        # Update QA description user selection
        tmp_selection = []
        for item in self.lwQAParamDesc.selectedItems():
            tmp_selection.append(item.text())

        self.user_qa_selection[current_qa_param] = tuple(tmp_selection)

        # There are pre selected items
        if len(self.user_qa_selection[current_qa_param]) > 0:
            # For each item selected by the user, make it selected
            # in the QA parameter description list view
            for qa_desc in self.user_qa_selection[current_qa_param]:
                item = self.lwQAParamDesc.findItems(qa_desc,
                        QtCore.Qt.MatchFixedString)

                item[0].setSelected(True)

        self.tbrSelection.setText(json.dumps(self.user_qa_selection,
             indent=2))
        self.lwQAParamDescUpdateInProgress == True

    @pyqtSlot()
    def update_qa_definition(self):
        """
        Updates the table view QA defintion and all the
        QA analytics widgets
        """
        # Clean QA user selection
        self.user_qa_selection = None

        # Fill QA definitions table view
        self.tvQADef.clearContents()
        self.fill_QA_definition_table()

        # Fill analytics data
        self.cmbQAParamName.clear()
        self.lwQAParamDesc.clear()
        self.fill_analytics()

    @pyqtSlot()
    def on_pbGetQADefinitions_click(self):
        """
        Creates a TATSSI Analytics object
        """
        # Wait cursor
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create the QA analytics object
        self.qa_analytics = Analytics(
                source_dir=self.lblDataDir.text(),
                product=self.txtProduct.toPlainText(),
                chunked=True,
                version=self.txtVersion.toPlainText(),
                start=self.start_date.text(),
                end=self.end_date.text())

        # Fill QA definition combo box
        self.cmbQADef.clear()
        qa_defs = []
        for i in range(len(self.qa_analytics.qa_defs)):
            qa_def = self.qa_analytics.qa_defs[i].QualityLayer.unique()[0]
            qa_defs.append(qa_def)

        self.cmbQADef.addItems(qa_defs)

        # Standard cursor
        QtWidgets.QApplication.restoreOverrideCursor()

        # Fill QA definitions table view
        #self.tvQADef.clearContents()
        #self.fill_QA_definition_table()

        # Fill analytics data
        #self.cmbQAParamName.clear()
        #self.lwQAParamDesc.clear()
        #self.fill_analytics()

    @pyqtSlot()
    def update_qa_param_def_description(self):
        """
        Update QA param definition descriptions
        """
        #sender = self.sender()
        # Continue only of the sender is not a QPushButton
        # and the cmbQADef has some text
        #if type(sender) == QtWidgets.QPushButton:
        #    return None
        #if len(sender.currentText()) == 0:
        #    return None

        self.lwQAParamDescUpdateInProgress = False
        self.lwQAParamDesc.clear()

        qa_def = self.cmbQADef.currentText()
        if len(qa_def) == 0:
            return None

        index = self.cmbQADef.findText(qa_def, QtCore.Qt.MatchFixedString)

        current_qa_param = self.cmbQAParamName.currentText()
        if len(current_qa_param) == 0:
            return None

        tmp_qa_param_def_description = self.qa_analytics.qa_defs[index]\
                [self.qa_analytics.qa_defs[index].Name == current_qa_param]

        # Ipdate description for QA parameter
        self.lwQAParamDesc.addItems(
                tmp_qa_param_def_description.Description.to_list())

        # Update QA selection user selection
        for qa_desc in self.user_qa_selection[current_qa_param]:
            item = self.lwQAParamDesc.findItems(qa_desc,
                    QtCore.Qt.MatchFixedString)

            item[0].setSelected(True)

        self.lwQAParamDescUpdateInProgress = True

    @pyqtSlot()
    def fill_analytics(self):
        """
        Fill QA data analytics cmbQAParamName and associated
        list widget with QA param definition descriptions
        """
        qa_def = self.cmbQADef.currentText()
        index = self.cmbQADef.findText(qa_def, QtCore.Qt.MatchFixedString)

        # Set user_qa_selection
        qa_params = self.qa_analytics.qa_defs[index].Name.unique().tolist()
        self.user_qa_selection = collections.OrderedDict(
                (element, '') for element in qa_params)
        # Set self.qa_analytics.qa_def
        self.qa_analytics.qa_def = self.qa_analytics.qa_defs[index]

        # Fill cmbQAParamName
        self.cmbQAParamName.addItems(qa_params)

        # Fill lwQAParamDesc
        self.update_qa_param_def_description()

    @pyqtSlot()
    def fill_QA_definition_table(self):
        """
        Fill the QA definition TableView based on the current QA def selection
        The QA def selection is obtained from the cmbQADef text
        """
        qa_def = self.cmbQADef.currentText()
        index = self.cmbQADef.findText(qa_def, QtCore.Qt.MatchFixedString)

        # Set number of entries
        rows, cols = self.qa_analytics.qa_defs[index].shape
        self.tvQADef.setRowCount(rows)
        self.tvQADef.setColumnCount(cols)

        for row in range(rows):
            for col in range(cols):
                # Insert item on QA def TableView
                item = str(self.qa_analytics.qa_defs[index].iloc[row, col])
                self.tvQADef.setItem(row, col,
                        QtWidgets.QTableWidgetItem(item))

        self.tvQADef.resizeColumnsToContents()

    @pyqtSlot()
    def on_pbOutputDirectory_click(self):
        """
        Opens dialog to select output dir and sets the
        lblDataDir label text
        """
        output_dir = open_file_dialog('directory')
        self.lblDataDir.setText(f'{output_dir}')

    @staticmethod
    def message_box(message_text):
        dialog = QtWidgets.QMessageBox()
        dialog.setIcon(QtWidgets.QMessageBox.Critical)
        dialog.setText(message_text)
        dialog.addButton(QtWidgets.QMessageBox.Ok)
        dialog.exec()

        return None

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

