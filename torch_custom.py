import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import csv


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MNIST(nn.Module):
    def __init__(self,input_size, num_classes):
        super(MNIST,self).__init__()
        self.fc1 = nn.Linear(input_size,15)
        self.fc2 = nn.Linear(15,30)
        self.fc3 = nn.Linear(30,10)
        self.fc4 = nn.Linear(10,num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x 

    def mask(self,model_tensor): 
        self.fc1.weight = torch.nn.parameter.Parameter(model_tensor['fc1']*self.fc1.weight)
        self.fc2.weight = torch.nn.parameter.Parameter(model_tensor['fc2']*self.fc2.weight)
        self.fc3.weight = torch.nn.parameter.Parameter(model_tensor['fc3']*self.fc3.weight)
        self.fc4.weight = torch.nn.parameter.Parameter(model_tensor['fc4']*self.fc4.weight)

class MNISTDataset(Dataset):
    def __init__(self, datapath,train,size):
        self.data = pd.read_csv(datapath)
        arr_data = np.array(self.data,dtype=np.float64)
        
        if train==True:
            arr_data = arr_data[:int(size*len(arr_data))]
        else : 
            arr_data = arr_data[int(size*len(arr_data)):]
        self.arr_data = (arr_data)
        self.x = torch.from_numpy(self.arr_data[:,1:])
        self.x = self.x.float()
        self.y = torch.from_numpy(self.arr_data[:,0])
        self.y = self.y.float()
       
    def __len__(self):
        return len(self.arr_data)

    def __getitem__(self,index):
        return self.x[index], F.one_hot(self.y[index].long(),10)


def model_summary(model):
    """
    get each layer weights, keys and shapes
    """
    assert isinstance(model,nn.Module)==True

    model_keys = []
    model_shapes = []
    for name, params in model.named_parameters():
        if name.split(".")[1]=="weight":
            model_keys.append(name.split(".")[0])
            model_shapes.append(params.shape[0]*params.shape[1])

    return model_keys,model_shapes



def get_weights(model:nn.Module,model_dict):
    """
    check whether model is instance of nn.Module
    """
    
    
    for name, param in model.named_parameters():
        if name.split(".")[1]=="weight":
            index = name.split(".")[0]
            temp_tensor = param.clone()
            temp_weights = temp_tensor.detach().cpu().numpy()
            temp_weights = temp_weights.reshape(-1,)
            
            model_dict[f'{index}'] = np.vstack((model_dict[f'{index}'],temp_weights))

    return model_dict




def save_weights(model:nn.Module,name):
    """
    check whether model is instance of nn.Module
    """
    W = []
    for layer, param in model.named_parameters():
        if layer.split(".")[1]=="weight":
            # index = name.split(".")[0]
            temp_tensor = param.clone()
            temp_weights = temp_tensor.detach().cpu().numpy()
            temp_weights = temp_weights.reshape(-1)
            W.extend(temp_weights.tolist())

    with open(name,'a',newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(W)
    file.close()
    

    



def get_accuracy(model:nn.Module, name,dataloader,device,save_weights = False):
    model.eval()
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for data,targets in dataloader:
            data = data.to(device)
            
            targets = targets.to(device)
            outputs = model(data)
            predictions = torch.argmax(outputs,1)
            targets = torch.argmax(targets,1)
            n_samples += targets.shape[0]
            n_correct += (predictions == targets).sum().tolist()

    acc = 100*n_correct/n_samples
    if save_weights:
        with open(name,'a',newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([acc])
        file.close()
    else:
        return acc
