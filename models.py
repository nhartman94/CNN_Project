"""
Models for electromagnetic particle shower classification. 
	(1) FC Network --> Softmax
	(2) 3 CNN --> FC network --> Softmax
	(3) 3 CNN --> CNN ---> FC network --> Softmax 
See Git Repo (https://github.com/nhartman94/CNN_Project) for more information.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

def flatten(x):
    N = x.shape[0] # read in N, H, W
    return x.view(N, -1)  # "flatten" the H * W values into a single vector per image
 


class FCNet(nn.Module):

    def __init__(self, inptDim=504, h1_dim=150, h2_dim=100, h3_dim=50, p=.5):

       '''
       FC net with 3 hidden layers, relu nonlinearities, batch-norm and dropout      

       Inputs:
 
       '''
       super().__init__()
       nOut = 3

       self.fc1 = nn.Linear(inptDim, h1_dim) 
       self.fc2 = nn.Linear(h1_dim, h2_dim)
       self.fc3 = nn.Linear(h2_dim, h3_dim)
       self.fc4 = nn.Linear(h3_dim, nOut)

       self.h1_dim = h1_dim
       self.h2_dim = h2_dim
       self.h3_dim = h3_dim

       self.dropout = p

       #self.modelName = "fc_{}_{}"

     
    def forward(self, layer0, layer1, layer2):

        # Flatten the inputs
        x = torch.cat((flatten(layer0),flatten(layer1),flatten(layer2)), dim=1) 

        # First hidden layer
        h1 = self.fc1(x)
        h1 = nn.BatchNorm1d(self.h1_dim)(h1)
        h1 = nn.ReLU()(h1)
        h1 = nn.Dropout(self.dropout)(h1)

        # Second hidden layer
        h2 = self.fc2(h1)
        h2 = nn.BatchNorm1d(self.h2_dim)(h2)
        h2 = nn.ReLU()(h2)
        h2 = nn.Dropout(self.dropout)(h2)

        # Third hidden layer
        # DON'T PUT ANY DROPOUT JUST BEFORE THE OUTPUT NODE
        h3 = self.fc3(h2)
        h3 = nn.BatchNorm1d(self.h3_dim)(h3)
        h3 = nn.ReLU()(h3)
       
        # To output classification
        scores = self.fc4(h3) 

        return scores

