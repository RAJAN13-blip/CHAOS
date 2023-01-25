import os
from data import *
from dense import Dense
import logging
from activations import *

def store_weights(network):
    stored_weights = []

    for items in network:
        if items.__class__.__name__ == 'Dense':
            stored_weights.append(items.weights)
            stored_weights.append(items.bias)

    stored_weights = np.asarray(stored_weights);

    return stored_weights

def initialize_Weights(network,start_weights):
    i =  0

    for items in network:
        if items.__class__.__name__ == 'Dense':
            items.weights = start_weights[i]
            items.bias = start_weights[i+1]
            i += 2
    
    return network

def parameter_file(data_name,path,initialization,new_init):
    
    dataset_path = os.path.join(path,"data")
    init_path = os.path.join(path,"initializations")
    
    
    if data_name == "vowel":

        network = [
        Dense(3,4), 
        Sigmoid(),
        Dense(4,6),  
        Softmax()
        ]   
        
        # 3-4-6
        
        if not new_init:
            init =  os.path.join(init_path,f"{data_name}\\{data_name}-{initialization}.npy")
            weights = np.load(init,allow_pickle=True)
            network = initialize_Weights(network,weights)
        else:
            weights = store_weights(network=network)
            network = initialize_Weights(network,weights)

        params = {}

        params['init'] = weights
        params['learning_rate'] = 0.2
        params['num_epochs'] = 50
        params['num_classes'] = 6
        params['num_features'] = 3
        params['loss'] = 'mse'
        params['data'] = vowel(dataset_path,params,0.8)
        params['network'] = network

      
        return params
    
    elif data_name == "iris":
        
        alpha = 0.05     # learning rate
        num_iterations = 200  # epochs
        init = 0.100   # Initial value of the chosen weight 
        init_words = 'point 1' # subdirectory name for init, create a dircetory by this name in the path folder
        perturb = 0.001  # perturbation to add to the weight
        lip_lr = False    # use lipshitz learning rate
        loss = 'mse' # 
        

        # 6-3/4-3
        return 

    elif data_name == "sonar":
        data = 'Sonar' # Vowel | Sonar | 
        path = "C:\\Users\\shiva\\chaos dyn\\Results\\Sonar\\"    # folder where the weights will be stored
        alpha = 0.05     # learning rate
        num_iterations = 150  # epochs
        init = 0.100   # Initial value of the chosen weight 
        init_words = 'point 1' # subdirectory name for init, create a dircetory by this name in the path folder
        perturb = 0.001  # perturbation to add to the weight
        lip_lr = False    # use lipshitz learning rate
        loss = 'mse'  
      
        # 60-4-1  
        return 


    elif data_name == "titanic":
        
      
        
        return 
    elif data_name == "cancer":
        
      
        
        return 

    elif data_name == "liver":
        
      
        
        return 

    
    