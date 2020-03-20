from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os



class LoadSave:

    @staticmethod
    def load_image(DIR) :
        img = Image.open(DIR)
        img.load()
        data = np.asarray( img, dtype="int32" )
        #data = data / 255
        #print(data.ndim)
        if data.ndim == 3:
            data = data[:, :, 0]
        #data = np.asarray(data).reshape(-1)
        t = torch.Tensor(data)
        t = t.reshape(1,480,640)
        t = t /255
        return t

    @staticmethod
    def save_image( t, outfilename , dimY) :
        t = t*255
        na = t.detach().numpy()
        na = na[0,:]
        npdata = np.asarray(na).reshape(dimY,-1) #480
        img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
        img.save(outfilename)

    @staticmethod
    def load_dataset(path,batchSize,numWorkers):
        data_path = path
        train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor(),
    )
        train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchSize,
        num_workers=numWorkers,
        shuffle=False
    )
        return train_loader

    @staticmethod
    def create_directory(DIR):
        try:
            os.mkdir('./IntertrainingDataLSTM/'+DIR)
        except OSError as e:
            print ("Creation of the directory %s failed" % DIR)
            print(e)
        else:
            print ("Successfully created the directory %s " % DIR)
