from torch_custom import * 
import bisect 


initialization = 1

name = f"weights{initialization}.csv"
train_acc = f"train_accuracies{initialization}.csv"
test_acc = f"test_accuracies{initialization}.csv"

datapath = r"mnist.csv"

model = MNIST(784,10).to(device) #change the model definition from the torch_custom file to test on other datasets
train_size = 0.7
datasets = MNISTDataset(datapath,train=True,size=train_size) #change the dataset class from the torch_custom file
test_datasets = MNISTDataset(datapath,train=False,size=train_size)
batch_size = 64
dataloader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True)
testloader = DataLoader(dataset=test_datasets,batch_size=batch_size)

num_epochs = 4
model_keys = []
model_shapes = []
learning_rate = 0.01


n_total_steps = len(dataloader)
print(n_total_steps)

torch.save(model.state_dict(), "model_1.pth")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)



model_tensors = {k:[] for k in model_keys}
mask_list = [1 ,2  , 398, 22, 24, 39200]

for name,params in model.named_parameters():
    if name.split(".")[1] == "weight":
        shape_ = params.shape
        model_tensors[f'{name.split(".")[0]}'] = torch.ones_like(params)

"""
TRAINING LOOP AND VALIDATION LOOP 
"""

for epoch in range(num_epochs):
    model.train()
    for i, (data, targets) in enumerate(dataloader):
        model.mask(model_tensors)
        data = data.to(device)
        targets = targets.float()
        targets.to(device)

        outputs = model(data)

        loss = criterion(outputs, targets)

        optimizer.step()

        get_accuracy(model=model,name =train_acc,dataloader=dataloader,save_weights=True)
        get_accuracy(model=model,name = test_acc,dataloader=testloader,save_weights=True)

    print(epoch,train_acc,get_accuracy(model=model,name =train_acc,dataloader=dataloader))
    print(epoch,test_acc,get_accuracy(model=model,name = test_acc,dataloader=testloader))