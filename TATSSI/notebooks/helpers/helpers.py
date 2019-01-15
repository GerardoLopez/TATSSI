
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

def OpenFileDialog():
    """
    Creates a Open File dialog window
    :return: Full path of selected file
    """
    app = QtWidgets.QApplication([dir])
    fname = QFileDialog.getOpenFileName(None, 
                                        "Select a file...",
                                        '.',
                                        filter="All files (*)")

    return str(fname[0])
