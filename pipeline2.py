from torch_custom import * 
import bisect 



model = MNIST(784,10)
model.load_state_dict("model_weights.pth")
model.to(device)

datapath = r"C:\\Users\\91993\\Desktop\\chaos\\NN-RESULTS-FINAL\\datasets\\data\\mnist_train.csv"

"""
MODEL INTITIALIZATION AND SETTING UP THE HYPERPARAMETERS
"""
model = MNIST(784,10).to(device)
train_size = 0.7
datasets = MNISTDataset(datapath,train=True,size=train_size)
test_datasets = MNISTDataset(datapath,train=False,size=train_size)

dataloader = DataLoader(dataset=datasets,batch_size=64,shuffle=True)
testloader = DataLoader(dataset=test_datasets,batch_size=64)

num_epochs = 5
model_keys = []
model_shapes = []
learning_rate = 0.1

model_keys, model_shapes = model_summary(model)
zipped = zip(model_keys,model_shapes)
model_dict = {f'{k}':np.arange(t) for k,t in zipped}
n_total_steps = len(dataloader)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

train_acc = []
test_acc = []


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

        train_acc.append(get_accuracy(model=model,dataloader=dataloader))
        test_acc.append(get_accuracy(model=model,dataloader=dataloader))

"""

"""


df_train = pd.DataFrame(train_acc)
df_test  = pd.DataFrame(test_acc)

df_train.to_csv("train_accuracies_sparse.csv",index=False,header=False)
df_test.to_csv("test_accuracies_sparse.csv",index=False,header=False)