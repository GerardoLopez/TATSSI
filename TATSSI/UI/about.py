
import os
import sys

# TATSSI modules
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = Path(current_dir).parents[1]
sys.path.append(str(src_dir.absolute()))

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSlot

class AboutUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(AboutUI, self).__init__(parent)
        uic.loadUi('about.ui', self)
        self.parent = parent

        self.show()

        # Connect methods with events
        self.pbHelpOK.helpRequested.connect(
                self.on_pbHelpOK_click)

    @pyqtSlot()
    def on_pbHelpOK_click(self):
        """
        Opens the corresponding TATSSI help file
        """
        # Set help file
        doc_dir = os.path.join(src_dir, 'doc')
        fname = 'TATSSI-v0.1-beta.1.pdf'

        # Open file
        try:
            cmd = f"cd {doc_dir} ; /usr/bin/xdg-open {fname}"
            os.system(cmd)
        except:
            message_text = (
                    f'Default application to open a PDF file '
                    f'is not set. You can access the file on:\n'
                    f'{fname}')
            message_box(message_text)
            return None

    @staticmethod
    def message_box(message_text):
        dialog = QtWidgets.QMessageBox()
        dialog.setIcon(QtWidgets.QMessageBox.Critical)
        dialog.setText(message_text)
        dialog.addButton(QtWidgets.QMessageBox.Ok)
        dialog.exec()

        return None


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DownloadersUI()
    app.exec_()
