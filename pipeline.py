from torch_custom import *
import tqdm 

initialization = 1

name = f"weights{initialization}.csv"
train_acc = f"train_accuracies{initialization}.csv"
test_acc = f"test_accuracies{initialization}.csv"

datapath = r"mnist.csv"

"""
MODEL INTITIALIZATION AND SETTING UP THE HYPERPARAMETERS

"""

model = MNIST(784,10).to(device) #change the model definition from the torch_custom file to test on other datasets
train_size = 0.7
datasets = MNISTDataset(datapath,train=True,size=train_size) #change the dataset class from the torch_custom file
test_datasets = MNISTDataset(datapath,train=False,size=train_size)
batch_size = 64
dataloader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True)
testloader = DataLoader(dataset=test_datasets,batch_size=batch_size)

num_epochs = 25
learning_rate = 0.05

n_total_steps = len(dataloader)
print(n_total_steps)

torch.save(model.state_dict(), "model_1.pth")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


"""
TRAINING AND VALIDATION LOOPS 
"""

for epoch in range(num_epochs):

 
    model.train()
    for i,(data,targets) in enumerate(dataloader):
        save_weights(model=model,name=name)
       
        data = data.to(device)
        temp_targets = targets
        targets = targets.float()

        targets = targets.to(device)
   
      
    
        outputs = model(data)
        loss = criterion(outputs,targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%1000==0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
        get_accuracy(model=model,name =train_acc,dataloader=dataloader,device=device,save_weights=True)
        get_accuracy(model=model,name = test_acc,dataloader=testloader,device=device,save_weights=True)

    print(epoch,train_acc,get_accuracy(model=model,name =train_acc,device=device,dataloader=dataloader))
    print(epoch,test_acc,get_accuracy(model=model,name = test_acc,device=device,dataloader=testloader))