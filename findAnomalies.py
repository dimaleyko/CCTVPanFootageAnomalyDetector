import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from LSImP import LoadSave
import os, os.path
import wandb
import numpy as np
import sys
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go

DIRLSTM = "models/LSTM2C.ckpt"
DIRSIAM = "models/Siamese02.ckpt"
DIRINA = "CombinedData/Anom"
DIRINNA = "CombinedData/NonAnom"
DIROUT = "CombinedOut/2"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")

loadsave = LoadSave()

chart_studio.tools.set_credentials_file(
    username="dimaleyko", api_key="xHe1gkvJ4LlsWx4Im0ta"
)

# Hyper-parameters
sequence_length = 480
input_size = 640
hidden_size = 100
num_layers = 4
num_classes = 307200
batch_size = 50
num_epochs = 1
learning_rate = 0.0001

n = 0

threshhold = 0.8
threshholdNA = 1 - threshhold + 0.05

values = np.zeros([10, 10])
valuesth = np.zeros(100)
valuesthA = np.zeros(100)
valuesTF = np.zeros(100)
valuesAF = np.zeros(100)
valuesFP = np.zeros(100)
valuesM = np.zeros(100)
valuesP = np.zeros(100)
valuesR = np.zeros(100)
valuesF = np.zeros(100)


class LSTM(nn.Module):
    def __init__(
        self, input_size, sequence_length, hidden_size, num_layers, num_classes
    ):
        super(LSTM, self).__init__()
        self.hidden_size0 = hidden_size
        self.num_layers0 = num_layers
        self.lstm0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc0 = nn.Linear(hidden_size, num_classes)

        self.hidden_size1 = hidden_size
        self.num_layers1 = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, xn):  # , xnn,xnnn, xnnnn
        # Set initial hidden and cell states
        # LSTM cell 1
        h0 = torch.zeros(self.num_layers0, x.size(0), self.hidden_size0).to(device)
        c0 = torch.zeros(self.num_layers0, x.size(0), self.hidden_size0).to(device)
        x = x.reshape(-1, sequence_length, input_size)
        # Forward propagate LSTM
        x, _ = self.lstm0(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = self.fc0(x[:, -1, :])
        x = x.reshape(1, 480, 640)

        x = (x + xn) / 2

        # LSTM cell 2
        h1 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(device)
        c1 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(device)
        x, _ = self.lstm1(
            x, (h1, c1)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        x = self.fc1(x[:, -1, :])
        x = x.reshape(1, 480, 640)

        out = x
        return out


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(10, 10), stride=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(2, 2)),
            nn.ReLU(),
        )
        self.liner = nn.Sequential(nn.Linear(7828, 3914), nn.ReLU())
        self.out = nn.Linear(3914, 2)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = torch.relu(self.liner(x))

        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        diff = torch.abs(out1 - out2)
        out = torch.sigmoid(self.out(diff))
        return out


def isAnomaly(predicted, original, i, th, thA, isAnom, totalFound, totalAnom):
    predicted = predicted.reshape(1, 1, 480, 640)
    original = original.reshape(1, 1, 480, 640)
    output = siamese(predicted, original)
    if output[0, 0] > th and output[0, 1] < (thA):
        print("Anomaly found in frame " + str(i))
        print(output)
        # loadsave.save_image(original.cpu(),DIROUT+'/output%06d.jpg' %(i),480)
        totalFound += 1
        if isAnom == True:
            totalAnom += 1
    return totalFound, totalAnom


def getPredFrames(i, th, thA, totalFound, totalAnom):
    if os.path.isfile(DIRINA + "/output%06d.jpg" % (i + 1)) == True:
        imageinN1 = loadsave.load_image(DIRINA + "/output%06d.jpg" % (i + 1)).to(device)
    else:
        imageinN1 = loadsave.load_image(DIRINNA + "/output%06d.jpg" % (i + 1)).to(
            device
        )

    if os.path.isfile(DIRINA + "/output%06d.jpg" % (i + 2)) == True:
        imageinN2 = loadsave.load_image(DIRINA + "/output%06d.jpg" % (i + 2)).to(device)
    else:
        imageinN2 = loadsave.load_image(DIRINNA + "/output%06d.jpg" % (i + 2)).to(
            device
        )

    if os.path.isfile(DIRINA + "/output%06d.jpg" % (i + 3)) == True:
        imageout = loadsave.load_image(DIRINA + "/output%06d.jpg" % (i + 3)).to(device)
        isAnom = True
    else:
        imageout = loadsave.load_image(DIRINNA + "/output%06d.jpg" % (i + 3)).to(device)
        isAnom = False

    outputs = lstm(imageinN1, imageinN2)
    loss = criterion(outputs, imageout)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    totalFound, totalAnom = isAnomaly(
        outputs, imageout, i + 3, th, thA, isAnom, totalFound, totalAnom
    )
    return totalFound, totalAnom


for i in range(10):
    for j in range(10):
        print("Doing: " + str(i + 1) + "&" + str(j + 1))
        th = threshhold - 0.01 * i
        thA = threshholdNA - 0.01 * j
        lstm = LSTM(
            input_size, sequence_length, hidden_size, num_layers, num_classes
        ).to(device)
        siamese = Siamese().to(device)
        lstm.load_state_dict(torch.load(DIRLSTM, map_location=device))
        siamese.load_state_dict(torch.load(DIRSIAM, map_location=device))
        lstm.to(device)
        siamese.to(device)
        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

        totalFound = 0
        totalAnom = 0

        for frame in range(113, 912):  # 113 912
            totalFound, totalAnom = getPredFrames(frame, th, thA, totalFound, totalAnom)
        # values[i, j,0] = th
        # values[i, j,1] = thA
        # values[i, j, 2] = totalFound
        # values[i, j, 3] = totalAnom
        # values[i, j, 4] = totalFound - totalAnom
        # values[i, j, 5] = 47 - totalAnom
        # values[i, j, 6] = totalAnom / totalFound
        # values[i, j, 7] = totalAnom / (totalAnom + (47 - totalAnom))
        # values[i, j, 8] = (
        #     2
        #     * (values[i, j, 5] * values[i, j, 6])
        #     / (values[i, j, 5] + values[i, j, 6])
        # )
        falsePositive = totalFound - totalAnom
        missed = 47 - totalAnom
        presision = totalAnom / totalFound
        recall = totalAnom / (totalAnom + (47 - totalAnom))
        f1 = 2 * (presision * recall) / (presision + recall)
        valuesth[i * 10 + j] = th
        valuesthA[i * 10 + j] = thA
        valuesTF[i * 10 + j] = totalFound
        valuesAF[i * 10 + j] = totalAnom
        valuesFP[i * 10 + j] = falsePositive
        valuesM[i * 10 + j] = missed
        valuesP[i * 10 + j] = presision
        valuesR[i * 10 + j] = recall
        valuesF[i * 10 + j] = f1
    # np.savetxt('DataNonAnom.csv', values, delimiter=',', fmt='%d')
dataTF = go.Mesh3d(x=valuesth, y=valuesthA, z=valuesTF, alphahull=5)

dataAF = go.Mesh3d(x=valuesth, y=valuesthA, z=valuesAF, alphahull=5)

dataFP = go.Mesh3d(x=valuesth, y=valuesthA, z=valuesFP, alphahull=5)

dataM = go.Mesh3d(x=valuesth, y=valuesthA, z=valuesM, alphahull=5)

dataP = go.Mesh3d(x=valuesth, y=valuesthA, z=valuesP, alphahull=5)

dataR = go.Mesh3d(x=valuesth, y=valuesthA, z=valuesR, alphahull=5)

dataF = go.Mesh3d(x=valuesth, y=valuesthA, z=valuesF, alphahull=5)


data = [dataTF, dataAF, dataFP, dataM, dataP, dataR, dataF]
py.iplot(data, filename="TestForThreshold")

