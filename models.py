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
 

"""
Fully connected net. Architecture: 
  Input --> Flatten --> [Linear --> BatchNorm --> ReLU] x N --> Softmax 
"""
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

        return scores1900

"""
Baseline CNN classifier architecture: 

  3 x ([Conv --> Batchnorm --> ReLU] x N) --> FC --> Softmax 

where the "3 x" indicates that there are three seperate CNN classifiers for each calorimeter layer image. 
These are then concatenated and fed into an FC network. 

For simplicity, the CNN layers of the initial model will follow this pattern: 

  [N - 1 CNN layers that preserve the input volume] --> [1 reduce layer that takes input to 3 x 6] --> FC 

"""
class ThreeCNN(nn.Module): 

  def __init__(self, layer0_params, layer0_reduce, layer1_params, layer1_reduce, layer2_params, layer2_reduce, fc_params):

    """
    Initializes a ThreeCNN object with the following inputs: 

      layer0_params = [# filters, filter_height, filter_width, stride, padding] --> paramaters that define the 
        volume-preserving layers of the CNN network for the 0th calorimeter image. 

      layer0_reduce = [same parameter types as above] --> parameters defining the reducing CNN layer. Note that 
        the parameters should be chosen so the output of this layer is of volume 3 x 6. 

    All other inputs follow the same pattern for the other calorimeter images, except for fc_params: 

      fc_params = [input_dimension, h1_dim, d2_dim, h3_dim, p]

    where input_dimension = 18*3 = 54 for this case, hi_dim is the number of dimensions in the ith hidden layer, and 
    p is the probability of keeping a given node during drouput. 

    """

    """
    These arrays hold all of the parameters for the ThreeCNN object.

      layer0_params_all[0] == # filters 
      ...
      layer0_params_all[5] == # filters in reducing CNN layer 
      ...
    """

    super().__init__()

    self.layer0_params_all = []
    self.layer1_params_all = []
    self.layer2_params_all = []
    self.fc_all = []

    # load volume-preserving layers 
    for i in range(4):
      self.layer0_params_all.append(layer0_params[i])
      self.layer1_params_all.append(layer1_params[i])
      self.layer2_params_all.append(layer2_params[i])

    # load reduction layers 
    for j in range(4):
      self.layer0_params_all.append(layer0_reduce[j])
      self.layer1_params_all.append(layer1_reduce[j])
      self.layer2_params_all.append(layer2_reduce[j])

    # load fc layer 
    for k in range(5):
      self.fc_all.append(fc_params[k])

  def forward_preprocess(self):
    """
    Forward pass for the three preprocessing CNN's. Returns: a tuple of models (0, 1, 2) corresponding to each 
    calorimeter image preprocessing forward pass. 

    Note that the outputs of this forward pass will be flattened/concatenated in the forward_fc function below. 
    """

    # CNN preprocessing for the 0th calorimeter image 
    # note that the input channel is 1 in this case
    layer0_model = nn.Sequential(
      nn.Conv2d(1, self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4]), 
      nn.BatchNorm2d(self.layer0_params_all[0]),
      nn.ReLU(), 
      nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4]), 
      nn.BatchNorm2d(self.layer0_params_all[0]),
      nn.ReLU(),
      nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4]), 
      nn.BatchNorm2d(self.layer0_params_all[0]),
      nn.ReLU(),
      nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[5], (self.layer0_params_all[6], self.layer0_params_all[7]), stride=self.layer0_params_all[8], padding=self.layer0_params_all[9]),
      nn.BatchNorm2d(self.layer0_params_all[5]), 
      nn.ReLU()
    )

    # CNN preprocessing for the 1st calorimeter image 
    layer1_model = nn.Sequential(
      nn.Conv2d(1, self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4]), 
      nn.BatchNorm2d(self.layer1_params_all[0]),
      nn.ReLU(), 
      nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4]), 
      nn.BatchNorm2d(self.layer1_params_all[0]),
      nn.ReLU(),
      nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4]), 
      nn.BatchNorm2d(self.layer1_params_all[0]),
      nn.ReLU(),
      nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[5], (self.layer1_params_all[6], self.layer1_params_all[7]), stride=self.layer1_params_all[8], padding=self.layer1_params_all[9]),
      nn.BatchNorm2d(self.layer1_params_all[5]), 
      nn.ReLU()
    )

     # CNN preprocessing for the 2nd calorimeter image 
    layer2_model = nn.Sequential(
      nn.Conv2d(1, self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4]), 
      nn.BatchNorm2d(self.layer2_params_all[0]),
      nn.ReLU(), 
      nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4]), 
      nn.BatchNorm2d(self.layer2_params_all[0]),
      nn.ReLU(),
      nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4]), 
      nn.BatchNorm2d(self.layer2_params_all[0]),
      nn.ReLU(),
      nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[5], (self.layer2_params_all[6], self.layer2_params_all[7]), stride=self.layer2_params_all[8], padding=self.layer2_params_all[9]),
      nn.BatchNorm2d(self.layer2_params_all[5]), 
      nn.ReLU()
    )

    return (layer0_model, layer1_model, layer2_model)

  def forward_fc(self):
    """
    Completes the FC layer forward pass. 
    Output: a model fc_model that computes class scores
    NOTE: assumes that the output tensors of the models defined in forward_preprocess have already been 
    flattened and concatenated together into a 1 x 54 vector. This will be completed in the training function. 
    """

    fc_model = nn.Sequential(
      nn.Linear(self.fc_all[0], self.fc_all[1]),
      nn.BatchNorm1d(self.fc_all[1]),
      nn.ReLU(),
      nn.Dropout(self.fc_all[4]),
      nn.Linear(self.fc_all[1], self.fc_all[2]),
      nn.BatchNorm1d(self.fc_all[2]),
      nn.ReLU(),
      nn.Dropout(self.fc_all[4]),
      nn.Linear(self.fc_all[2], self.fc_all[3]),
      nn.BatchNorm1d(self.fc_all[3]),
      nn.ReLU(),
      nn.Dropout(self.fc_all[4])
    )

    return fc_model
