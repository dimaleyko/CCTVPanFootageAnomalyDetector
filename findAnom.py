import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from LSImP import LoadSave
import os, os.path

loadsave = LoadSave()

DIRO = 'AnomTest/Original'
DIRP = 'AnomTest/Predicted'
DIRS = 'AnomTest/Test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GPU')
else:
    print('CPU')
criterion = torch.nn.L1Loss()





def wholePictureLossAnom():
    loss = criterion(imageO,imageP)
    return loss
def oneKernelPictureLossAnom():
    for i in range(640):
        for j in range(480):
            loss = criterion(imageO[0][j][i],imageP[0][j][i])
            if loss > 0.05:
                #print (loss)
                imageO[0][j][i] = 0
    loadsave.save_image(imageO, DIRS+'/outTest.jpg', 480)

def nKernelPictureLossAnom(n,file, imageO, imageP):
    x = 640 - (n-1)
    y = 480 - (n-1)
    imageC = imageO.to(device)
    for j in range(int(0+(n-1)/2),int(480-(n-1)/2)):
        for i in range(int(0+(n-1)/2),int(640-(n-1)/2)):
            loss = criterion(getKernelValues(n,i,j,imageO),getKernelValues(n,i,j,imageP))
            #print (loss)
            if loss > 0.4:
                imageC[0][j][i] = 0
    loadsave.save_image(imageO.cpu(), DIRS+'/outTest|'+str(n)+'|'+str(file-3410)+'|.jpg', 480)
    print('done')

def getKernelValues(n,i,j,image):
    kernel = torch.zeros((n,n), dtype=torch.float64).to(device)
    x = int(j - (n-1)/2)
    y = int(i - (n-1)/2)
    for xj in range(n):
        for yi in range(n):
            #print(str(x+xj)+'|'+str(y+yi))
            kernel[xj][yi] = image[0][x+xj][y+yi]
    return kernel
total = 0
# for file in range(3425,3475):
def my_func(file):
    imageO = loadsave.load_image(DIRO+'/output%06d.jpg' %(file)).to(device)
    imageP = loadsave.load_image(DIRP+'/output%06d.jpg' %(file)).to(device)
    nKernelPictureLossAnom(3,file,imageO,imageP)
    """loss = wholePictureLossAnom()
    if loss > 0.48:
        total += 1
        print(str(loss)+'|'+str(file)+'|'+str(total))
    if file == 4:
        print(str(loss)+'|'+str(file)+'|'+str(total))"""

import multiprocessing
multiprocessing.Pool().map(my_func, range(3425,3475))