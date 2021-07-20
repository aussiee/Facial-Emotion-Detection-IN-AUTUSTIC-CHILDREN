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

def Calc_Wt(TRR,TST):
        WTRN = TRR
        M = []
        ERR = []
        for i in range(0, WTRN.shape[1]):
            #print(i,'\n')
            RR = WTRN[:,i]
            WTST =TST[:]
            #print('TRAIN:1 \n',np.shape(RR))
            #print('TEST:1 \n',np.shape(WTST))
            Temp = np.subtract(WTST, RR)
            ERR = LA.norm(Temp)
            #print(ERR)
            M.append(ERR)
        ind = np.argmin(M);
        MM=np.min(M)
        ind=np.floor(ind/10)+1;
        return ind,MM
    

    
