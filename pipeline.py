from torch_custom import *
import tqdm 

initialization = 1

datapath = r"C:\\Users\\91993\\Desktop\\chaos\\NN-RESULTS-FINAL\\datasets\\data\\mnist_train.csv"

"""
MODEL INTITIALIZATION AND SETTING UP THE HYPERPARAMETERS
"""
model = MNIST(784,10).to(device) #change the model definition from the torch_custom file to test on other datasets
train_size = 0.7
datasets = MNISTDataset(datapath,train=True,size=train_size) #change the dataset class from the torch_custom file
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

"""
TRAINING AND VALIDATION LOOPS 
"""

for epoch in range(num_epochs):

 
    model.train()
    for i,(data,targets) in enumerate(dataloader):
        model_dict = get_weights(model=model,model_dict=model_dict)
       
        data = data.to(device)
        temp_targets = targets
        targets = F.one_hot(targets.long(),num_classes=10)
        targets = targets.float()

        targets = targets.to(device)
   
      
    
        outputs = model(data)
        loss = criterion(outputs,targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc.append(get_accuracy(model=model,dataloader=dataloader))
        test_acc.append(get_accuracy(model=model,dataloader=dataloader))

        if(i+1)%100==0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')




"""
SAVE WEIGHT PROFILES 
"""


df  = pd.DataFrame(model_dict['fc1'])
df = df.iloc[1:]


df2 = pd.DataFrame(model_dict['fc2'])
df2 = df2.iloc[1:]

final_df = pd.concat([df,df2],axis=1)
new_axis = [k for k in range(final_df.shape[1])]
final_df.set_axis(new_axis, axis = 1, inplace=True)

final_df.to_csv(f"weights{initialization}.csv")

"""
1. store ( intitial) weights at the beginning
2. store train_acc, store test_acc
"""

df_train = pd.DataFrame(train_acc)
df_test  = pd.DataFrame(test_acc)

df_train.to_csv("train_accuracies_normal.csv",index=False,header=False)
df_test.to_csv("test_accuracies_normal.csv",index=False,header=False)