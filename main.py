from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys
from mysql.connector import connect, Error

import math
import os
import pickle
import tarfile
import time

import cv2
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision import transforms as trans
from data_gen import data_transforms



def FaceToVector(image):
    device = torch.device("cpu")
    checkpoint = 'BEST_checkpoint_r18.tar'
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()
    img0 = cv2.imread(image)
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transform2 = trans.Compose([
        trans.Resize([int(112), int(112)]),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    resized_image = cv2.resize(img0, (112, 112))

    tensor1 = test_transform(resized_image).to(device).unsqueeze(0)
    start_time = time.time()
    output = model(tensor1)

    feature0 = output[0].detach().numpy()
    x0 = feature0 / np.linalg.norm(feature0)
    x1 = x0.tolist()
    return str(x1)




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(284, 290)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)

        self.LayoutAdd = QtWidgets.QGridLayout()
        self.gridLayout.addLayout(self.LayoutAdd, 2, 0, 1, 1)

        self.DB_Layout = QtWidgets.QFormLayout()
        self.gridLayout.addLayout(self.DB_Layout, 0, 0, 1, 2)

        self.labelHost = QtWidgets.QLabel(self.centralwidget)
        self.DB_Layout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelHost)

        self.lineHost = QtWidgets.QLineEdit(self.centralwidget)
        self.lineHost.setText('localhost')
        self.DB_Layout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineHost)

        self.labelUser = QtWidgets.QLabel(self.centralwidget)
        self.DB_Layout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.labelUser)

        self.lineUser = QtWidgets.QLineEdit(self.centralwidget)
        self.DB_Layout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineUser)

        self.labelPass = QtWidgets.QLabel(self.centralwidget)
        self.DB_Layout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.labelPass)

        self.linePass = QtWidgets.QLineEdit(self.centralwidget)
        self.linePass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.DB_Layout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.linePass)

        self.labelDB = QtWidgets.QLabel(self.centralwidget)
        self.DB_Layout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.labelDB)

        self.lineDB = QtWidgets.QLineEdit(self.centralwidget)
        self.DB_Layout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineDB)

        self.ndicatorDB = QtWidgets.QLabel(self.centralwidget)
        self.gridLayout.addWidget(self.ndicatorDB, 1, 0, 1, 1)

        self.ConnectButton = QtWidgets.QPushButton(self.centralwidget)
        self.ConnectButton.setObjectName("ConnectButton")
        self.ConnectButton.clicked.connect(self.connect_database)
        self.gridLayout.addWidget(self.ConnectButton, 0, 2, 1, 1)

        #image and name selction layout

        self.labelName = QtWidgets.QLabel(self.centralwidget)
        self.LayoutAdd.addWidget(self.labelName, 2, 0, 1, 1)

        self.lineName = QtWidgets.QLineEdit(self.centralwidget)
        self.lineName.setEnabled(False)
        self.LayoutAdd.addWidget(self.lineName, 2, 1, 1, 1)

        self.lineFace = QtWidgets.QLineEdit(self.centralwidget)
        self.lineFace.setEnabled(False)
        self.LayoutAdd.addWidget(self.lineFace, 0, 1, 1, 1)

        self.labelFace = QtWidgets.QLabel(self.centralwidget)
        self.LayoutAdd.addWidget(self.labelFace, 0, 0, 1, 1)

        self.AddButton = QtWidgets.QPushButton(self.centralwidget)
        self.gridLayout.addWidget(self.AddButton, 3, 0, 1, 1)
        self.AddButton.setEnabled(False)
        self.AddButton.clicked.connect(self.addFaceInDB)

        self.Add_label = QtWidgets.QLabel(self.centralwidget)
        self.gridLayout.addWidget(self.Add_label, 2, 2, 1, 1 )

        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.LayoutAdd.addWidget(self.toolButton, 0, 2, 2, 1)
        self.toolButton.setEnabled(False)
        self.toolButton.clicked.connect(self.openFileFace)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 290, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle("Addition vector of face to the DB")
        self.labelName.setText(_translate("MainWindow", "name"))
        self.labelFace.setText(_translate("MainWindow", "face"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.labelHost.setText(_translate("MainWindow", "host"))
        self.labelUser.setText(_translate("MainWindow", "user"))
        self.labelPass.setText(_translate("MainWindow", "password"))
        self.labelDB.setText(_translate("MainWindow", "database"))
        self.ConnectButton.setText(_translate("MainWindow", "Connect"))
        self.AddButton.setText(_translate("MainWindow", "Add"))

    def getFace(self):
        return self.lineFace.text()

    def openFileFace(self):
        fileFace = QFileDialog.getOpenFileNames(self, "Выберите картинку с лицами фантомов", "*.png; *.jpg")[0]
        #self.textEdit.setText(data)
        self.lineFace.setText(str(fileFace[0]))

    def connect_database(self):
        global conn
        try:
            conn = connect(host=self.lineHost.text(),
                           user=self.lineUser.text(),  # input("Имя пользователя: )"
                           password=self.linePass.text(),  # getpass("Пароль ")
                           database=self.lineDB.text(),
                           auth_plugin='mysql_native_password')
            self.ndicatorDB.setText("CORRECT\n" + str(conn))  # print("CORRECT",conn) #input("DB: ")
            self.lineFace.setEnabled(True)
            self.lineName.setEnabled(True)
            self.toolButton.setEnabled(True)
            self.AddButton.setEnabled(True)
        except Error as e:
            self.ndicatorDB.setText("Connection DB error\n" + str(e))

    def addFaceInDB(self):
        insert_knownFace_query = "INSERT INTO known_faces (id, name, vector) VALUES ( %s, %s, %s )"
        selectLastId = "SELECT id FROM known_faces ORDER BY id DESC LIMIT 1" #select last ID in DB
        cursor = conn.cursor()
        cursor.execute(selectLastId)
        _id = 0
        for row in cursor:
            _id = row[0]+1
        _name = self.lineName.text()
        _vector = FaceToVector(self.lineFace.text())
        knownFace_records = (_id, _name, _vector)
        try:
            cursor.execute(insert_knownFace_query, knownFace_records)
            conn.commit()
            self.Add_label.setText("CORRECT addition")
        except Error as er:
            self.Add_label.setText(str(er))




class ExampleApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # запускаем приложение

