import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from LSIm import LoadSave
import os, os.path

loadsave = LoadSave()

DIR = 'Data/cutGreyRevTest'+'/1'
print (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GPU')
else:
    print('CPU')

i = 0

im0 = loadsave.load_image('/output%06d.jpg' %(i+1)).to(device)
im1 = loadsave.load_image('/output%06d.jpg' %(i+2)).to(device)

imAv = (im0+im1)/2

loadsave.save_image(imAv.cpu(),'Avtest',480)
