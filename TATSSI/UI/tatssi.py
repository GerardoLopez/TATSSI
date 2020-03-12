
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot

# TATSSI UI dialogs
from downloaders import DownloadersUI

class TATSSI_UI(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(TATSSI_UI, self).__init__()
        uic.loadUi('tatssi.ui', self)

        # Connect actions
        self.actionDownloaders.triggered.connect(self.downloaders)

        self.show()

    def downloaders(self):
        """
        Launch the downloaders dialog
        """
        # Downloaders
        self.downloaders = DownloadersUI(parent=self)

if __name__ == "__main__":
    app = QApplication([])
    window = TATSSI_UI()
    app.exec_()
