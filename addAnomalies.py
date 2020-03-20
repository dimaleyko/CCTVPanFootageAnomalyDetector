import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from LSImP import LoadSave
import os, os.path
import random

ls = LoadSave()

DIRA = 'CombinedData/Anom'
DIRANOM = 'SiameseData/Anomalies'

# for i in range(900,1782): #1781
#     tn = ls.load_image(DIRA+'/output%06d.jpg' %(i+1))
#     r = random.randint(1,18)
#     ta = ls.load_image(DIRANOM+'/%01d.png' %(r))
#     x = random.randrange(319)
#     y = random.randrange(239)
#     for xi in range(320):
#         for yj in range(240):
#             if ta[0,xi,yj] == 0:
#                 tn[0,y+yj,x+xi] = tn[0,y+yj,x+xi]/3
#     ls.save_image(tn, DIRA+'/output%06d.jpg' %(i+1), 480)
#     print(i)





def my_func(i):
    if os.path.isfile(DIRA+'/output%06d.jpg' %(i+1)) == True:
        tn = ls.load_image(DIRA+'/output%06d.jpg' %(i+1))
        r = random.randint(1,18)
        ta = ls.load_image(DIRANOM+'/%01d.png' %(r))
        x = random.randrange(319)
        y = random.randrange(239)
        for xi in range(320):
            for yj in range(240):
                if ta[0,xi,yj] == 0:
                    tn[0,y+yj,x+xi] = tn[0,y+yj,x+xi]/3
        ls.save_image(tn, DIRA+'/output%06d.jpg' %(i+1), 480)
        print(i)

import multiprocessing
multiprocessing.Pool().map(my_func, range(113,915))