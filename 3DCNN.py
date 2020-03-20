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


ls = LoadSave()

class NeuralNetworkCalculator(nn.Module):
    def __init__(self):
        super(NeuralNetworkCalculator, self).__init__()
        self.layer_1 = nn.Conv3d(1, 1, kernel_size = (1,15,15), stride = (1,5,5)) #kernel = 5 padding = 2
        self.layer_2 = nn.Conv3d(1, 1, kernel_size = (1,10,10), stride = (1,3,3)) #kernel = 5 padding = 2
        self.layer_3 = nn.Conv3d(1, 1, kernel_size = (1,7,7), stride = (1,1,1)) #kernel = 5 padding = 2
        self.layer_4 = nn.Conv3d(1, 1, kernel_size = (1,5,9), stride = (1,1,1)) #kernel = 5 padding = 2
        self.layer_5 = nn.Conv3d(1, 1, kernel_size = (2,5,9), stride = (1,1,1)) #kernel = 5 padding = 2
        self.layer_6 = nn.Conv3d(1, 1, kernel_size = (1,5,7), stride = (1,1,1)) #kernel = 5 padding = 2
        self.layer_7 = nn.Conv3d(1, 1, kernel_size = (1,5,5), stride = (1,1,1)) #kernel = 5 padding = 2 
        self.layer_8 = nn.Conv3d(1, 1, kernel_size = (1,7,6), stride = (1,1,1)) #kernel = 5 padding = 2
        # self.layer_7 = nn.Linear(231, 2)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        x = F.relu(self.layer_6(x))
        # x = x.view(231)
        x = F.relu(self.layer_7(x))
        x = F.relu(self.layer_8(x))
        x = x.view((2))
        return x



if torch.cuda.is_available():
    print('GPU')
else:
    print('CPU')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

isAnomaly: torch.tensor

model = NeuralNetworkCalculator().to(device)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.001)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)



for i in range(3,3603):
    tp = ls.load_image(DIRP+'/output%06d.jpg' %(i+1)).to(device)
    if os.path.isfile(DIRA+'/output%06d.jpg' %(i+1)) == True:
        ta = ls.load_image(DIRA+'/output%06d.jpg' %(i+1)).to(device)
        isAnomaly = torch.tensor([1,0]).float().to(device)
    else:
        ta = ls.load_image(DIRNA+'/output%06d.jpg' %(i+1)).to(device)
        isAnomaly = torch.tensor([0,1]).float().to(device)
    t = torch.cat((tp,ta))

    t = t.view(1,1,2,480,640)

    optimizer.zero_grad()
    output = model(t)
    loss = criterion(output,isAnomaly)
    loss.backward()
    optimizer.step()
    if i%100 == 0:
        print(t)
        print(loss)
        print(output)
        print(isAnomaly)