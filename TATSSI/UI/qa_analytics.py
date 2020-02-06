
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from TATSSI.input_output.utils import *
from TATSSI.notebooks.helpers.utils import *
from TATSSI.notebooks.helpers.qa_analytics import Analytics
from TATSSI.qa.EOS.catalogue import Catalogue

import ogr
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import pyqtSlot

from TATSSI.UI.helpers.utils import *

class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('qa_analytics.ui', self)

        # Set attributes
        self.qa_analytics = None

        # Connect methods with events
        self.pbOutputDirectory.clicked.connect(
                self.on_pbOutputDirectory_click)

        self.pbGetQADefinitions.clicked.connect(
                self.on_pbGetQADefinitions_click)

        self.cmbQAParamName.currentIndexChanged.connect(
                self.update_qa_param_def_description)

        self.cmbQADef.currentIndexChanged.connect(
                self.update_qa_definition)

        # Set default date display format
        self.start_date.setDisplayFormat('dd-MM-yyyy')
        self.end_date.setDisplayFormat('dd-MM-yyyy')

        self.show()

    def update_qa_definition(self):
        """
        Updates the table view QA defintion and all the
        QA analytics widgets
        """
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

        # Fill QA definitions table view
        self.tvQADef.clearContents()
        self.fill_QA_definition_table()

        # Fill analytics data
        self.cmbQAParamName.clear()
        self.lwQAParamDesc.clear()
        self.fill_analytics()

    @pyqtSlot()
    def update_qa_param_def_description(self):
        """
        Update QA param definition descriptions
        """
        self.lwQAParamDesc.clear()

        qa_def = self.cmbQADef.currentText()
        index = self.cmbQADef.findText(qa_def, QtCore.Qt.MatchFixedString)

        current_qa_param = self.cmbQAParamName.currentText()
        tmp_qa_param_def_description = self.qa_analytics.qa_defs[index]\
                [self.qa_analytics.qa_defs[index].Name == current_qa_param]

        self.lwQAParamDesc.addItems(
                tmp_qa_param_def_description.Description.to_list())

    @pyqtSlot()
    def fill_analytics(self):
        """
        Fill QA data analytics cmbQAParamName and associated
        list widget with QA param definition descriptions
        """
        qa_def = self.cmbQADef.currentText()
        index = self.cmbQADef.findText(qa_def, QtCore.Qt.MatchFixedString)

        # Fill cmbQAParamName
        qa_params = self.qa_analytics.qa_defs[index].Name.unique().tolist()
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

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

