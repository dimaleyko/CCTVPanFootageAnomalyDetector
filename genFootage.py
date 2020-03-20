import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from LSImP import LoadSave
import os, os.path

loadsave = LoadSave()

DIRT = 'SiameseData/PreLSTM/Validate'
DIRO = 'SiameseData/PredictedLSTM/3'
MODELDIR = "models/LSTM2C.ckpt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GPU')
else:
    print('CPU')

# Hyper-parameters
sequence_length = 480
input_size = 640
hidden_size = 100
num_layers = 4
num_classes = 307200
batch_size = 50
num_epochs = 1
learning_rate = 0.0001

class RNN(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size0 = hidden_size
        self.num_layers0 = num_layers
        self.lstm0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc0 = nn.Linear(hidden_size, num_classes)

        self.hidden_size1 = hidden_size
        self.num_layers1 = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)


    def forward(self, x, xn): #, xnn,xnnn, xnnnn
        # Set initial hidden and cell states
        #LSTM cell 1
        h0 = torch.zeros(self.num_layers0, x.size(0), self.hidden_size0).to(device)
        c0 = torch.zeros(self.num_layers0, x.size(0), self.hidden_size0).to(device)
        x = x.reshape(-1,sequence_length, input_size)
        # Forward propagate LSTM
        x, _ = self.lstm0(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = self.fc0(x[:, -1, :])
        x = x.reshape(1,480,640)

        x = (x+xn)/2

        #LSTM cell 2
        h1 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(device)
        c1 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(device)
        x, _ = self.lstm1(x, (h1, c1))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        x = self.fc1(x[:, -1, :])
        x = x.reshape(1,480,640)


        out = x
        return out

def getPredFrames():
    for file in range(1002):
        imageinN1 = loadsave.load_image(DIRT+'/output%06d.jpg' %(file+1)).to(device)
        imageinN2 = loadsave.load_image(DIRT+'/output%06d.jpg' %(file+2)).to(device)
        imageout = loadsave.load_image(DIRT+'/output%06d.jpg' %(file+3)).to(device)
        outputs = model(imageinN1,imageinN2)
        loadsave.save_image(outputs.cpu(),DIRO+'/output%06d.jpg' %(file+3),480)
        loss = criterion(outputs, imageout)

    # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(file)
        #print(outputs)

model = RNN(input_size,sequence_length, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(MODELDIR, map_location=device))
model.to(device)
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
getPredFrames()
