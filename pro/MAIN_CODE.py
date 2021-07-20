from __future__ import print_function

import tkinter
from tkinter import filedialog
from tkinter import messagebox 

import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

from PyQt4 import QtCore, QtGui


import math
import random
import string
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from array import array
from numpy import linalg as LA
from Sim_SV import Calc_Wt
import time

def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('TRUE CLASS')
    plt.xlabel('PREDICTED CLASS')
    plt.tight_layout()

    
root=tkinter.Tk()
print('******Start*****')
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow1(object):
    

    def setupUii(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1200, 800)
        MainWindow.setStyleSheet(_fromUtf8("\n""background-image: url(bg3.jpg);\n"""))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(750, 180, 111, 27))
        self.pushButton.clicked.connect(self.quit)
        self.pushButton.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
       
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
#################################################################
        

        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 180, 131, 27))
        self.pushButton_2.clicked.connect(self.show1)
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        
        self.pushButton_4 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(550, 220, 131, 27))
        self.pushButton_4.clicked.connect(self.show2)
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        
        

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
       
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Machine Attack ", None))
        self.pushButton_2.setText(_translate("MainWindow", "TEST", None))
        self.pushButton_4.setText(_translate("MainWindow", "TRAIN", None))
        self.pushButton.setText(_translate("MainWindow", "Exit", None))

    def quit(self):
        print ('Process end')
        print ('******End******')
        quit()
         
    def show1(self):
        image_path= filedialog.askopenfilename(filetypes = (("BROWSE  IMAGE", "*.jpg"), ("All files", "*")))
        img=cv2.imread(image_path)
        #img=cv2.imread('TEST1/A (5).png')
        cv2.imshow('FACE IMAGE',img)
        cv2.waitKey(300)
        cv2.destroyAllWindows()

        # GRAY CONVERSION
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('GRAY IMAGE',img)
        cv2.waitKey(300)
        cv2.destroyAllWindows()
            
        # RESIZING
        img = cv2.resize(img,(224,224),1)
        cv2.imshow('RESIZED IMAGE',img)
        cv2.waitKey(300)
        cv2.destroyAllWindows()

        # MEDIAN FILTERED
        ROI= cv2.medianBlur(img,5)
        cv2.imshow('MEDIAN IMAGE',ROI)
        cv2.waitKey(300)
        cv2.destroyAllWindows()

        # ---------------------------------------------------------------------------
        # FEATURE EXTRACTION
        # 1]GABOR FEATURE EXTRACTION
        filters = build_filters()
        GF1 = process(ROI, filters)
        cv2.imshow('Extracted Texture',GF1)
        cv2.waitKey(300)
        cv2.destroyAllWindows()

        FV1=np.array(np.sum(GF1, axis = 0))
        FV2=np.array(np.sum(GF1, axis = 1))


        FV=np.concatenate([FV1[:],FV2[:]])
        FV=np.array(FV)

        print('Image features: \n',np.shape(FV))
        Img=FV

        print('Image features: \n',np.shape(Img))


        import pickle
        # LOAD 
        file= open("Gnet.cnn",'rb')
        TRR = pickle.load(file)
        file.close()


        IND,MM=Calc_Wt(TRR,Img)
        if IND==1:
            print('ANGRY\n')
        elif IND==2:
            print('HAPPY\n')
        elif IND==3:
            print('SAD\n')


        
        
    def show2(self):
        file= open("TRNMDL.obj",'rb')
        cnf_matrix = pickle.load(file)
        file.close()
        plt.figure()
        plot_confusion_matrix(cnf_matrix[0:2,0:2], classes=['Emotion','Normal'], normalize=True,title='Proposed Detection')
        plt.show()





if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUii(MainWindow)
    MainWindow.move(550, 170)
    MainWindow.show()
    sys.exit(app.exec_())
    

