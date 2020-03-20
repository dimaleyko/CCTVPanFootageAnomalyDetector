import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from LSImP import LoadSave
import os, os.path
import wandb
import sys


DIRA = 'SiameseData/Anom'
DIRNA = 'SiameseData/NonAnom'
DIRP = 'SiameseData/Predicted'
DIRAV = 'SiameseData/AnomV'
DIRNAV = 'SiameseData/NonAnomV'
DIRPV = 'SiameseData/PredictedV'
DIR = 'Siamese'

losses = torch.tensor([0.00007,0.0000675,0.000065,0.0000625,0.00006,0.0000575,0.000055,0.0000525,0.00005,0.0000475,0.000045,0.0000425,0.00004,0.0000375,0.000035,0.0000325,0.00003])
gss = torch.tensor([1,0.75,0.5,0.25])

ls = LoadSave()
num_epochs = 5
frame_total = 0
learning_rate = losses.numpy()[int(sys.argv[1])]
scheduler_gamma = gss.numpy()[int(sys.argv[2])]
OUTNAME = 'Siamese'+ sys.argv[1]+sys.argv[2] +str(num_epochs)





class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size = (10,10), stride = (3,3)),
            nn.ReLU(), 
            nn.Conv2d(1, 1, kernel_size = (7,7), stride = (2,2)),
            nn.ReLU(),
        )
        self.liner = nn.Sequential(
            nn.Linear(7828, 3914), 
            nn.ReLU()
            )
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


if torch.cuda.is_available():
    print('GPU')
else:
    print('CPU')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# isAnomaly: torch.tensor
# isAB: bool

model = Siamese().to(device)

os.system("wandb login 860b947e5e349701cef1a2ebcd28bb89150bd7d5")

wandb.init(project="pan-footage-siamese", name = 'lr = '+str(learning_rate)+' sg = '+str(scheduler_gamma)+' F '+str(num_epochs))

wandb.watch(model)

def train(i):
    tp = ls.load_image(DIRPF+'/output%06d.jpg' %(i+1)).to(device)
    if os.path.isfile(DIRAF+'/output%06d.jpg' %(i+1)) == True:
        t = ls.load_image(DIRAF+'/output%06d.jpg' %(i+1)).to(device)
        isAnomaly = torch.tensor([[1,0]]).float().to(device)
    else:
        t = ls.load_image(DIRNAF+'/output%06d.jpg' %(i+1)).to(device)
        isAnomaly = torch.tensor([[0,1]]).float().to(device)
    tp = tp.view(1,1,480,640)
    t = t.view(1,1,480,640)

    optimizer.zero_grad()
    output = model.forward(tp,t)

    loss = criterion(output,isAnomaly)
    loss.backward()
    optimizer.step()
    return loss, output, isAnomaly

def validate():
    correctGuesses = 0
    tottalLoss = 0
    model.eval()
    datasamplesV = len([name for name in os.listdir(DIRPV) if os.path.isdir(os.path.join(DIRPV, name))])
    for folderV in range(datasamplesV):
        DIRPVF = DIRPV+'/'+str(folderV+1)
        DIRAVF = DIRAV+'/'+str(folderV+1)
        DIRNAVF = DIRNAV+'/'+str(folderV+1)
        framesV = len([name for name in os.listdir(DIRPVF) if os.path.isfile(os.path.join(DIRPVF, name))])
        for i in range(2,framesV):
            tp = ls.load_image(DIRPVF+'/output%06d.jpg' %(i+1)).to(device)
            if os.path.isfile(DIRAVF+'/output%06d.jpg' %(i+1)) == True:
                t = ls.load_image(DIRAVF+'/output%06d.jpg' %(i+1)).to(device)
                isAnomaly = torch.tensor([[1,0]]).float().to(device)
                isAB = True
            else:
                t = ls.load_image(DIRNAVF+'/output%06d.jpg' %(i+1)).to(device)
                isAnomaly = torch.tensor([[0,1]]).float().to(device)
                isAB = False
            tp = tp.view(1,1,480,640)
            t = t.view(1,1,480,640)
            output = model.forward(tp,t)
            loss = criterion(output,isAnomaly)
            tottalLoss = tottalLoss + loss

            if (1-loss) > 0.75:
                correctGuesses += 1
    correctPersentage = (correctGuesses/framesV)*100
    model.train()
    avLoss = tottalLoss/framesV
    
    return correctPersentage, avLoss



criterion = torch.nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

for epoch in range(num_epochs):
    datasamples = len([name for name in os.listdir(DIRP) if os.path.isdir(os.path.join(DIRP, name))])
    for folder in range(datasamples):
        DIRPF = DIRP+'/'+str(folder+1)
        DIRAF = DIRA+'/'+str(folder+1)
        DIRNAF = DIRNA+'/'+str(folder+1)
        frames = len([name for name in os.listdir(DIRPF) if os.path.isfile(os.path.join(DIRPF, name))])-1
        for i in range(2,frames):
            lossT, output, isAnomaly = train(i)
            frame_total += 1
            if i%100 == 0:
                correctP, lossV = validate()

                print ('Epoch [{}/{}], Loss: {}, Actual: {}, Output: {}, Guesses: {}'.format(epoch+1, num_epochs, lossT.item(), isAnomaly.data.cpu().numpy(), output.data.cpu().numpy(), correctP))
                wandb.log({"Epoch": epoch+1,"Dataset": datasamples+1, "Frames in Folder": frames+1, "Validation Loss": lossV, "Train Loss": lossT, "Frames": frame_total, "Correct Guessed": correctP})
        scheduler.step()
    
torch.save(model.state_dict(), 'models/'+OUTNAME+'.ckpt')