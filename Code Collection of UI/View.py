# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'View.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(525, 450)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(130, 70, 256, 256))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("multiview/Blank.png"))
        self.label.setObjectName("label")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(150, 340, 231, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.confirmButton = QtWidgets.QPushButton(self.centralwidget)
        self.confirmButton.setGeometry(QtCore.QRect(220, 380, 80, 30))
        self.confirmButton.setIconSize(QtCore.QSize(20, 20))
        self.confirmButton.setObjectName("confirmButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 525, 21))
        self.menubar.setObjectName("menubar")
        self.menuOperations = QtWidgets.QMenu(self.menubar)
        self.menuOperations.setObjectName("menuOperations")
        self.menuBasic_Objects = QtWidgets.QMenu(self.menubar)
        self.menuBasic_Objects.setObjectName("menuBasic_Objects")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionCube = QtWidgets.QAction(MainWindow)
        self.actionCube.setObjectName("actionCube")
        self.actionCylinder = QtWidgets.QAction(MainWindow)
        self.actionCylinder.setObjectName("actionCylinder")
        self.actionSketch = QtWidgets.QAction(MainWindow)
        self.actionSketch.setObjectName("actionSketch")
        self.menuOperations.addAction(self.actionOpen)
        self.menuOperations.addAction(self.actionClose)
        self.menuBasic_Objects.addAction(self.actionCube)
        self.menuBasic_Objects.addAction(self.actionCylinder)
        self.menubar.addAction(self.menuOperations.menuAction())
        self.menubar.addAction(self.menuBasic_Objects.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "NavigationView"))
        self.confirmButton.setText(_translate("MainWindow", "Select"))
        self.menuOperations.setTitle(_translate("MainWindow", "File"))
        self.menuBasic_Objects.setTitle(_translate("MainWindow", "Basic Objects"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionCube.setText(_translate("MainWindow", "Cube"))
        self.actionCylinder.setText(_translate("MainWindow", "Cylinder"))
        self.actionSketch.setText(_translate("MainWindow", "Sketch"))
