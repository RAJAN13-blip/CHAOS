import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MNIST(nn.Module):
    def __init__(self,input_size, num_classes):
        super(MNIST,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x 

    def mask(self,model_tensor): 
        self.fc1.weight = torch.nn.parameter.Parameter(model_tensor['fc1']*self.fc1.weight)
        self.fc2.weight = torch.nn.parameter.Parameter(model_tensor['fc2']*self.fc2.weight)

class MNISTDataset(Dataset):
    def __init__(self, datapath,train,size):
        self.data = pd.read_csv(datapath)
        arr_data = np.array(self.data,dtype=np.float64)
        np.random.shuffle(arr_data)
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

def get_accuracy(model:nn.Module, dataloader):
    model.eval()
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for data,targets in dataloader:
            data = data.to(device)
            temp_targets = targets
            targets = F.one_hot(targets.long(),num_classes=10)
            targets = targets.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs,1)
            n_samples += targets.shape[0]
            n_correct += (predictions == temp_targets).float().sum()

    acc = 100*n_correct/n_samples
    return acc
