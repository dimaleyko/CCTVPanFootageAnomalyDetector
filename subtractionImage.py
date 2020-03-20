import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from LSImP import LoadSave
import os, os.path

DIRA = 'AnomTest/Anom'
DIRNA = 'AnomTest/NonAnom'
DIRP = 'AnomTest/Predicted'
DIRSA = 'AnomTest/AMPA'
DIRS = 'AnomTest/AMP'

ls = LoadSave()


def my_func(i):
    tp = ls.load_image(DIRP+'/output%06d.jpg' %(i+1))
    if os.path.isfile(DIRA+'/output%06d.jpg' %(i+1)) == True:
        ta = ls.load_image(DIRA+'/output%06d.jpg' %(i+1))
        td1 = ta - tp
        ls.save_image(abs(td1), DIRSA+'/output%06d.jpg' %(i+1), 480)
    else:
        ta = ls.load_image(DIRNA+'/output%06d.jpg' %(i+1))
        td1 = ta - tp
        ls.save_image(abs(td1), DIRS+'/output%06d.jpg' %(i+1), 480)

import multiprocessing
multiprocessing.Pool().map(my_func, range(3,12965))