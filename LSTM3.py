import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from LSImP import LoadSave
import os, os.path
import wandb
import numpy as np



loadsave = LoadSave()


# Device configuration
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
learning_rate = 0.00001
DIRT = 'LSTMData/Train'
DIRV = 'LSTMData/Validate'
OUTNAME = 'LSTM3C'
frame_total = 0

# MNIST dataset
#train_loader = loadsave.load_dataset('./Data/cutGreyRevTrain',1,1)
#test_loader = loadsave.load_dataset('./Data/cutGreyRevTest',1,1)

# Recurrent neural network (many-to-one)
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

        self.hidden_size2 = hidden_size
        self.num_layers2 = num_layers
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        """
        self.hidden_size3 = hidden_size
        self.num_layers3 = num_layers
        self.lstm3 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        self.hidden_size4 = hidden_size
        self.num_layers4 = num_layers
        self.lstm4 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc4 = nn.Linear(hidden_size, num_classes)"""

    def forward(self, x, xn, xnn): #, xnn,xnnn, xnnnn
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

        x = (x+xnn)/2
        #LSTM cell 3
        h2 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(device)
        c2 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(device)
        x, _ = self.lstm2(x, (h2, c2))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        x = self.fc2(x[:, -1, :])
        x = x.reshape(1,480,640)
        """
        x = (x+xnnn)/2
        #LSTM cell 4
        h3 = torch.zeros(self.num_layers3, x.size(0), self.hidden_size3).to(device)
        c3 = torch.zeros(self.num_layers3, x.size(0), self.hidden_size3).to(device)
        x, _ = self.lstm3(x, (h3, c3))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        x = self.fc3(x[:, -1, :])
        x = x.reshape(1,480,640)

        x = (x+xnnnn)/2
        #LSTM cell 5
        h4 = torch.zeros(self.num_layers4, x.size(0), self.hidden_size4).to(device)
        c4 = torch.zeros(self.num_layers4, x.size(0), self.hidden_size4).to(device)
        x, _ = self.lstm4(x, (h3, c3))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        x = self.fc4(x[:, -1, :])
        x = x.reshape(1,480,640)"""

        out = x
        return out

def evaluate(epoch, dataset, frames, trainloss, totalframes):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        totalLoss = 0.0
        averageLoss = 0.0
        total_step = len([name for name in os.listdir(DIRV) if os.path.isfile(os.path.join(DIRV, name))])-6
        for file in range(10):
            imageinN1 = loadsave.load_image(DIRV+'/output%06d.jpg' %(file+1)).to(device)
            imageinN2 = loadsave.load_image(DIRV+'/output%06d.jpg' %(file+2)).to(device)
            imageinN3 = loadsave.load_image(DIRV+'/output%06d.jpg' %(file+3)).to(device)
            """imageinN4 = loadsave.load_image(DIRV+'/output%06d.jpg' %(file+4)).to(device)
            imageinN5 = loadsave.load_image(DIRV+'/output%06d.jpg' %(file+5)).to(device)"""
            imageout = loadsave.load_image(DIRV+'/output%06d.jpg' %(file+4)).to(device)


            outputs = model(imageinN1,imageinN2, imageinN3)#,imageinN3, imageinN4, imageinN5
            loss = criterion(outputs, imageout)
            totalLoss = totalLoss + loss
        averageLoss = totalLoss/total_step
        wandb.log({"Epoch": epoch+1,"Dataset": dataset+1, "Frames in Folder": frames+1, "Test Loss": loss, "Train Loss": trainloss, "Frames": totalframes})

model = RNN(input_size,sequence_length, hidden_size, num_layers, num_classes).to(device)
os.system("wandb login 860b947e5e349701cef1a2ebcd28bb89150bd7d5")

wandb.init(project="pan-footage-lstm")

wandb.watch(model)



# Loss and optimizer
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

if os.path.isfile('IntertrainingDataLSTM/'+OUTNAME) != True:
    loadsave.create_directory(OUTNAME)

# Train the model
for epoch in range(num_epochs):
    datasamples = len([name for name in os.listdir(DIRT) if os.path.isdir(os.path.join(DIRT, name))])
    for folder in range(datasamples):
        
        DIRRT = DIRT+'/'+str(folder+1)
        total_step = len([name for name in os.listdir(DIRRT) if os.path.isfile(os.path.join(DIRRT, name))])-6
        for file in range(total_step):
            imageinN1 = loadsave.load_image(DIRRT+'/output%06d.jpg' %(file+1)).to(device)
            imageinN2 = loadsave.load_image(DIRRT+'/output%06d.jpg' %(file+2)).to(device)
            imageinN3 = loadsave.load_image(DIRRT+'/output%06d.jpg' %(file+3)).to(device)
            """imageinN4 = loadsave.load_image(DIRRT+'/output%06d.jpg' %(file+4)).to(device)
            imageinN5 = loadsave.load_image(DIRRT+'/output%06d.jpg' %(file+5)).to(device)"""
            imageout = loadsave.load_image(DIRRT+'/output%06d.jpg' %(file+4)).to(device)

            frame_total = frame_total + 1



            outputs = model(imageinN1, imageinN2,imageinN3)#, imageinN3, imageinN4, imageinN5
            loss = criterion(outputs, imageout)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (file+1) % int(total_step/4) == 0 or (file+1) % (total_step-1) == 0: #7932
                print ('Epoch [{}/{}], Folder [{}/{}], File[{}/{}], Loss: {}, Learning Rate: {} '.format(epoch+1, num_epochs, folder+1, datasamples, file+1, total_step, loss.item(), scheduler.get_lr()))
                loadsave.save_image(outputs.cpu(),'IntertrainingDataLSTM/'+OUTNAME+'/output'+str(file+2)+'|'+str(folder+1)+'|'+str(epoch)+'.png',480)
            elif (file + 1) % 100 == 0:
                evaluate(epoch, folder,file,loss.item(), frame_total)
                model.train()
    scheduler.step()

torch.save(model.state_dict(), 'models/'+OUTNAME+'.ckpt')
np.save("weights3", weights)
wandb.save("weights3.npy")
