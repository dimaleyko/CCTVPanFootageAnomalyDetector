import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from LSImP import LoadSave
import os, os.path
import random

ls = LoadSave()

DIRO = 'CombinedData/1'
DIRA = 'CombinedData/Anom'
DIRNA = 'CombinedData/NonAnom'

# for i in range(3603):
#     t = ls.load_image(DIRO+'/output%06d.jpg' %(i+1))
#     r = random.randint(0,1)
#     if r == 0:
#         ls.save_image(t, DIRNA+'/output%06d.jpg' %(i+1),480)
#     else:
#         ls.save_image(t, DIRA+'/output%06d.jpg' %(a+1),480)
        

def blah(i):
    t = ls.load_image(DIRO+'/output%06d.jpg' %(i+1))
    r = random.randint(0,802)
    if r > 40:
        ls.save_image(t, DIRNA+'/output%06d.jpg' %(i+1),480)
    else:
        ls.save_image(t, DIRA+'/output%06d.jpg' %(i+1),480)

import multiprocessing
multiprocessing.Pool().map(blah, range(113,915))