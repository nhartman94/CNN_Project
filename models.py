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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 

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

       self.bn1 = nn.BatchNorm1d(h1_dim)
       self.bn2 = nn.BatchNorm1d(h2_dim)
       self.bn3 = nn.BatchNorm1d(h3_dim)

       self.dropout = p

       self.modelName = "fc_{}_{}_{}_dpt_{}".format(h1_dim, h2_dim, h3_dim,p)

     
    def forward(self, layer0, layer1, layer2):

        # Flatten the inputs
        x = torch.cat((flatten(layer0),flatten(layer1),flatten(layer2)), dim=1) 

        # First hidden layer
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = nn.ReLU()(h1)
        h1 = nn.Dropout(self.dropout)(h1)

        # Second hidden layer
        h2 = self.fc2(h1)
        h2 = self.bn2(h2) 
        h2 = nn.ReLU()(h2)
        h2 = nn.Dropout(self.dropout)(h2)

        # Third hidden layer
        # DON'T PUT ANY DROPOUT JUST BEFORE THE OUTPUT NODE
        h3 = self.fc3(h2)
        h3 = self.bn3(h3)
        h3 = nn.ReLU()(h3)
       
        # To output classification
        scores = self.fc4(h3) 

        return scores

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

      fc_params = [input_dimension, h1_dim, d2_dim, h3_dim, out_dimension, p]

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
    for i in range(5):
      self.layer0_params_all.append(layer0_params[i])
      self.layer1_params_all.append(layer1_params[i])
      self.layer2_params_all.append(layer2_params[i])


    # IMPORTANT: duplicate values not appended to list. 
    # need to fix this! 
    # load reduction layers 
    for j in range(5):
      self.layer0_params_all.append(layer0_reduce[j])
      self.layer1_params_all.append(layer1_reduce[j])
      self.layer2_params_all.append(layer2_reduce[j])

    # load fc layer 
    for k in range(6):
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
      nn.Dropout(self.fc_all[5]),
      nn.Linear(self.fc_all[1], self.fc_all[2]),
      nn.BatchNorm1d(self.fc_all[2]),
      nn.ReLU(),
      nn.Dropout(self.fc_all[5]),
      nn.Linear(self.fc_all[2], self.fc_all[3]),
      nn.BatchNorm1d(self.fc_all[3]),
      nn.ReLU(),
      nn.Dropout(self.fc_all[5]),
      nn.Linear(self.fc_all[3], self.fc_all[4])
    )

    return fc_model

class ThreeCNN_Module(nn.Module):
    
  def __init__(self, layer0_params, layer0_reduce, layer1_params, layer1_reduce, layer2_params, layer2_reduce, fc_params, layer3_params=[], flag=False):

    """
    Identical to the class ThreeCNN above, but uses the module API for easier integration with already-written training 
    code in train.py. 

    Initializes a ThreeCNN object with the following inputs: 

      layer0_params = [# filters, filter_height, filter_width, stride, padding] --> paramaters that define the 
        volume-preserving layers of the CNN network for the 0th calorimeter image. 

      layer0_reduce = [same parameter types as above] --> parameters defining the reducing CNN layer. Note that 
        the parameters should be chosen so the output of this layer is of volume 3 x 6. 

    All other inputs follow the same pattern for the other calorimeter images, except for fc_params: 

      fc_params = [input_dimension, h1_dim, d2_dim, h3_dim, out_dimension, p]

    where input_dimension = 18*3 = 54 for this case, hi_dim is the number of dimensions in the ith hidden layer, and 
    p is the probability of keeping a given node during drouput.
    
    The "flag" input, if set to True, places an extra convolutional layer [Conv --> Batchnorm --> ReLU] before flattening and
    the fully connected layer. It's default value is False. 
    Similarly, layer3_params contains the parameters for the extra processig layer between the three seperate CNN layers and 
    the final fc layer. 

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
    self.flag = flag
    if self.flag: 
        # load CNN preprocessing layer 
        self.layer3_params_all = layer3_params
    self.fc_all = [] 

    # load volume-preserving layers 
    for i in range(5):
      self.layer0_params_all.append(layer0_params[i])
      self.layer1_params_all.append(layer1_params[i])
      self.layer2_params_all.append(layer2_params[i])

    # load reduction layers 
    for j in range(5):
      self.layer0_params_all.append(layer0_reduce[j])
      self.layer1_params_all.append(layer1_reduce[j])
      self.layer2_params_all.append(layer2_reduce[j])

    # load fc layer 
    for k in range(6):
      self.fc_all.append(fc_params[k])
    
    # CNN parameters 
    # indexing: cnn_[calorimeter index]_[conv layer number]
    self.cnn_0_1 = nn.Conv2d(1, self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4])
    self.cnn_0_2 = nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4])
    self.cnn_0_3 = nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4])
    self.cnn_0_4 = nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[5], (self.layer0_params_all[6], self.layer0_params_all[7]), stride=self.layer0_params_all[8], padding=self.layer0_params_all[9])

    self.cnn_1_1 = nn.Conv2d(1, self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4])
    self.cnn_1_2 = nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4])
    self.cnn_1_3 = nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4])
    self.cnn_1_4 = nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[5], (self.layer1_params_all[6], self.layer1_params_all[7]), stride=self.layer1_params_all[8], padding=self.layer1_params_all[9])

    self.cnn_2_1 = nn.Conv2d(1, self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4])
    self.cnn_2_2 = nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4])
    self.cnn_2_3 = nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4])
    self.cnn_2_4 = nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[5], (self.layer2_params_all[6], self.layer2_params_all[7]), stride=self.layer2_params_all[8], padding=self.layer2_params_all[9])
    
    # optional CNN preprocessing layer
    # initially, this layer will also be volume-preserving 
    if self.flag: 
        self.cnn_3_1 = nn.Conv2d(3, self.layer3_params_all[0], (self.layer3_params_all[1], self.layer3_params_all[2]), stride=self.layer3_params_all[3], padding=self.layer3_params_all[4])
        
    # batchnorm parameters, convolutional layers 
    self.batch_0_1 = nn.BatchNorm2d(self.layer0_params_all[0], track_running_stats=False)
    self.batch_0_2 = nn.BatchNorm2d(self.layer0_params_all[5], track_running_stats=False)

    self.batch_1_1 = nn.BatchNorm2d(self.layer1_params_all[0], track_running_stats=False)
    self.batch_1_2 = nn.BatchNorm2d(self.layer1_params_all[5], track_running_stats=False)

    self.batch_2_1 = nn.BatchNorm2d(self.layer2_params_all[0], track_running_stats=False)
    self.batch_2_2 = nn.BatchNorm2d(self.layer2_params_all[5], track_running_stats=False)

    if self.flag: 
        self.batch_3_1 = nn.BatchNorm2d(self.layer3_params_all[0], track_running_stats=False)

    # batchnorm parameters, fc layers 
    self.batch_fc_1 = nn.BatchNorm1d(self.fc_all[1], track_running_stats=False)
    self.batch_fc_2 = nn.BatchNorm1d(self.fc_all[2], track_running_stats=False)
    self.batch_fc_3 = nn.BatchNorm1d(self.fc_all[3], track_running_stats=False)

    # FC parameters 
    self.lin_1 = nn.Linear(self.fc_all[0], self.fc_all[1])
    self.lin_2 = nn.Linear(self.fc_all[1], self.fc_all[2])
    self.lin_3 = nn.Linear(self.fc_all[2], self.fc_all[3])
    self.lin_final = nn.Linear(self.fc_all[3], self.fc_all[4])
    
    self.modelName = "model" 

  def forward(self, l0, l1, l2):

    """
    Forward pass for the network. Unlike in ThreeCNN, we use here the module API. 
    """

    # CNN forward pass for the 0th calorimeter layer image (input: l0)
    cnn_0 = self.cnn_0_1(l0) 
    cnn_0 = self.batch_0_1(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_2(cnn_0) 
    cnn_0 = self.batch_0_1(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_3(cnn_0)
    cnn_0 = self.batch_0_1(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_4(cnn_0)
    cnn_0 = self.batch_0_2(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)

    # CNN forward pass for the 1st calorimeter layer image (input: l1)
    cnn_1 = self.cnn_1_1(l1) 
    cnn_1 = self.batch_1_1(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_2(cnn_1) 
    cnn_1 = self.batch_1_1(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_3(cnn_1)
    cnn_1 = self.batch_1_1(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_4(cnn_1)
    cnn_1 = self.batch_1_2(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)

    # CNN forward pass for the 2nd calorimeter layer image (input: l2)
    cnn_2 = self.cnn_2_1(l2) 
    cnn_2 = self.batch_2_1(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_2(cnn_2) 
    cnn_2 = self.batch_2_1(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_3(cnn_2)
    cnn_2 = self.batch_2_1(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_4(cnn_2)
    cnn_2 = self.batch_2_2(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    
    # optional CNN preprocessing 
    if self.flag: 
        """
        Unsqueezing unecessary, because the shape of each cnn_i activation map from the above 
        is (batch_size, 1, 3, 6). So we can just concatenate along dimension 1 
        cnn_0 = torch.unsqueeze(cnn_0, 0) 
        cnn_1 = torch.unsqueeze(cnn_1, 0) 
        cnn_2 = torch.unsqueeze(cnn_2, 0) 
        """
        # x.shape = (3, 3, 6) 
        x = torch.cat((cnn_0, cnn_1, cnn_2), dim=1) 
        
        # preprocessing CNN layer 
        cnn_3 = self.cnn_3_1(x) 
        cnn_3 = self.batch_3_1(cnn_3) 
        cnn_3 = nn.ReLU()(cnn_3) 

    # flatten, concatenate outputs from CNN forward passes / preprocess  
    if self.flag: 
        x = flatten(cnn_3) 
    else: 
        x = torch.cat((flatten(cnn_0),flatten(cnn_1),flatten(cnn_2)), dim=1)

    # fully connected net forward pass 
    fc = self.lin_1(x) 
    fc = self.batch_fc_1(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    fc = self.lin_2(fc) 
    fc = self.batch_fc_2(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    fc = self.lin_3(fc) 
    fc = self.batch_fc_3(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    scores = self.lin_final(fc) 
  
    return scores

#########################################

'''
<<<<<<< Updated upstream
These two sequential models cast the layers as 12x12 images, used as global functions 
for the 3D CNN below.
=======
These sequential models cast the layers as 12x12 images
>>>>>>> Stashed changes
'''
layer0_12x12 = lambda nF1,nF2: nn.Sequential( nn.Conv2d(1,nF1, (1,8), stride=(1,8)),
                                              nn.ReLU(),
                                              nn.ConvTranspose2d(nF1,nF2, (4,1), stride=(4,1)),
                                              nn.ReLU()
                                           )

layer1_12x12 = lambda nF: nn.Sequential(nn.Conv2d(1,nF,(3,3),stride=1,padding=1))
    
layer2_12x12 = lambda nF: nn.Sequential(nn.ConvTranspose2d(1,nF, (1,2), stride=(1,2)),
                                  nn.ReLU())

########################################

# class inception(nn.Module):
# 
#     def __init__(self):
# 
#         self_conv_3x3   = nn.Conv2d(1,nF,(3,3),stride=(1,1),padding=(1,1))
#         self_conv_5x5   = nn.Conv2d(1,nF,(3,3),stride=(1,1),padding=(1,1))
#         self_conv_1x1_1 = nn.Conv2d(1,nF,(3,3),stride=(1,1),padding=(1,1))
#         self_conv_1x1_2 = nn.Conv2d(1,nF,(3,3),stride=(1,1),padding=(1,1))
#         self_conv_1x1_3 = nn.Conv2d(1,nF,(3,3),stride=(1,1),padding=(1,1))
#         self_conv_1x1_4 = nn.Conv2d(1,nF,(3,3),stride=(1,1),padding=(1,1))
# 
#     def forward(self, x):
# 
#         cnn_1x1_1 =  self.conv_1x1_1(x) 
#         cnn_1x1_2 =  self.conv_1x1_2(x)
#         pool_3x3  =  nn.MaxPool2d((3,3),stride=1, padding=1) 
# 
#         cnn_1x1_3 = self.conv_1x1_3(x)
#         cnn_3x3   = self.conv_3x3(x)
#         cnn_5x5   = self.conv_3x3(x)
#         cnn_1x1_4 = self.conv_1x1_4(pool_3x3)
#  
#         out = torch.cat((cnn_1x1_3, cnn_3x3, cnn_5x5, cnn_1x1_4),dim=?)


'''
These sequential models downsample the img to 3x6 
'''
layer0_3x6 = lambda nF: nn.Sequential(nn.Conv2d(1,nF,(3,3),stride=(1,1),padding=(1,1)),
                                      nn.BatchNorm2d(nF),
                                      nn.ReLU(),
                                      nn.MaxPool2d((1,2)),
                                      nn.Conv2d(nF,nF,(3,3),stride=(1,1),padding=(1,1)),
                                      nn.BatchNorm2d(nF),
                                      nn.ReLU(),
                                      nn.MaxPool2d((1,2)),
                                      nn.Conv2d(nF,nF,(3,3),stride=(1,1),padding=(1,1)),
                                      nn.BatchNorm2d(nF),
                                      nn.ReLU(),
                                      nn.MaxPool2d((1,2)),
                                      nn.Conv2d(nF,nF,(3,3),stride=(1,1),padding=(1,1)),
                                      nn.BatchNorm2d(nF),
                                      nn.ReLU(),
                                      nn.MaxPool2d((1,2))
                                     )

layer1_3x6 = lambda nF: nn.Sequential(nn.Conv2d(1,nF,(3,3),padding=(1,1)),
                                      nn.BatchNorm2d(nF),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,2)),
                                      nn.Conv2d(nF,nF,(3,3),padding=(1,1)),
                                      nn.BatchNorm2d(nF),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,1)),
                                     )

layer2_3x6 = lambda nF: nn.Sequential(nn.Conv2d(1,nF,(3,3),stride=(1,1),padding=(1,1)),
                                      nn.BatchNorm2d(nF),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,1)),
                                      nn.Conv2d(nF,nF,(3,3),padding=(1,1)),
                                      nn.BatchNorm2d(nF),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,1)),
                                     )

class CNN_3d(nn.Module):

    '''
    This first transforms the model input layers into a 12x12 dim images, and then
    applies a 3d convolution to the inputs  

    '''
    def __init__(self, spatialDim=12, preConvParams={'nF1':4, 'nF2':8}, 
                 nFilters_1 = 16, filter_1=(3,4,4), stride_1=(2,2,2), padding_1=(1,1,1),
                 nFilters_2 = 8,  filter_2=(2,2,2), stride_2=(1,2,2), padding_2=(0,1,1),
                 h1_dim=50, h2_dim=25, p=0.5):

       super().__init__()


       if isinstance(spatialDim, int) and spatialDim == 12:

           nF1 = preConvParams['nF1']
           nF2 = preConvParams['nF2']

           self.layer0_preConv = layer0_12x12(nF1,nF2)
           self.layer1_preConv = layer1_12x12(nF2)
           self.layer2_preConv = layer2_12x12(nF2)

           self.img_phi = 12 
           self.img_eta = 12 

           self.nF = nF2

       elif isinstance(spatialDim, tuple) and spatialDim[0] == 3 and spatialDim[1] == 6:

           nF = preConvParams['nF']

           self.layer0_preConv = layer0_3x6(nF) 
           self.layer1_preConv = layer1_3x6(nF) 
           self.layer2_preConv = layer2_3x6(nF) 

           self.img_phi = 3 
           self.img_eta = 6 

           self.nF = nF

       nOut = 3
       
       self.cnn3d_1 = nn.Conv3d(self.nF, nFilters_1, filter_1, stride_1, padding_1)
       self.bn3d_1 = nn.BatchNorm3d(nFilters_1)

       # Calculate the number of input dimensions seen by each of the inputs
       d1_out = (3 - filter_1[0] + 2*padding_1[0]) / stride_1[0] + 1 
       h1_out = (self.img_phi - filter_1[1] + 2*padding_1[1]) / stride_1[1] + 1
       w1_out = (self.img_eta - filter_1[2] + 2*padding_1[2]) / stride_1[2] + 1

       print("Output size after the first conv: {},{},{},{}".format(nFilters_1, d1_out, h1_out, w1_out))

       self.cnn3d_2 = nn.Conv3d(nFilters_1, nFilters_2, filter_2, stride_2, padding_2)
       self.bn3d_2 = nn.BatchNorm3d(nFilters_2)

       d2_out = (d1_out - filter_2[0] + 2*padding_2[0]) / stride_2[0] + 1 
       h2_out = (h1_out - filter_2[1] + 2*padding_2[1]) / stride_2[1] + 1
       w2_out = (w1_out - filter_2[2] + 2*padding_2[2]) / stride_2[2] + 1
       print("Output size after the second conv: {},{},{},{}".format(nFilters_2, d2_out, h2_out, w2_out))

       # After the 3d convolutions, flatten and classify the output
       fc_inpt = nFilters_2 * d2_out * h2_out * w2_out 

       self.fc1 = nn.Linear(fc_inpt, h1_dim) 
       self.fc2 = nn.Linear(h1_dim, h2_dim) 
       self.fc3 = nn.Linear(h2_dim, nOut) #h3_dim) 

       self.bn1 = nn.BatchNorm1d(h1_dim)
       self.bn2 = nn.BatchNorm1d(h2_dim)
 
       self.dropout = p

       self.modelName = "cnn3d_{}x{}_C{}_F{}{}{}_S{}{}{}_P{}{}{}_C{}_F{}{}{}_S{}{}{}_P{}{}{}_fc_{}_{}_dpt_{}".format(self.img_phi,self.img_eta,\
                        nFilters_1,*filter_1,*stride_1,*padding_1,nFilters_2,*filter_2,*stride_2,*padding_2,h1_dim,h2_dim,p)

    def forward(self, layer0, layer1, layer2):

        # Call the functions above to make the input dim of the three layers the same 
        l0 = self.layer0_preConv(layer0).view(-1, self.nF, 1, self.img_phi, self.img_eta)
        l1 = self.layer1_preConv(layer1).view(-1, self.nF, 1, self.img_phi, self.img_eta)
        l2 = self.layer2_preConv(layer2).view(-1, self.nF, 1, self.img_phi, self.img_eta)

        # Concatenate the inputs
        # Pytorch's 3d conv expects an input with shape (N, C_{in}, D, H, W)
        x = torch.cat((l0, l1, l2),dim=2)

        # First 3d conv layer
        cnn3d_1 = self.cnn3d_1(x)
        bn3d_1 = self.bn3d_1(cnn3d_1)

        # Second 3d conv layer
        cnn3d_2 = self.cnn3d_2(bn3d_1)
        bn3d_2 = self.bn3d_2(cnn3d_2)

        # Flatten the input
        y = flatten(bn3d_2)

        # First fc layer
        y = self.fc1(y)
        y = self.bn1(y)
        y = nn.ReLU()(y)
        y = nn.Dropout(self.dropout)(y)

        # Second fc layer
        y = self.fc2(y)
        y = self.bn2(y)
        y = nn.ReLU()(y)

        # Output scores
        scores = self.fc3(y)            

        return scores 
 


class rnn_2dCNN(nn.Module):

  """
  Recurrent two-dimensional CNN model. This model is designed to process the calorimeter data sequentially, with layer 0 
  at the earliest time and layer 2 at the latest time. 
  Model Architecture: 
  
     3 x ([Conv2d --> Batchnorm --> ReLU] x N) --> RNN --> [Conv1d --> Batchnorm --> ReLU] x M --> fc --> scores 

  The output of the CNN (after concatenation) will be of size (batch_size, 3, 3, 6). We'll shift this to (3, batch_size, 3*6), 
  and input this to the RNN, which will ouput a hidden layer tensor of size (num_layers, batch_size, hidden_size). This hidden 
  layer tensor will be input into another one-dimensional CNN processing layer, and then flattened/concatenated for 
  input to the fc layer. 

  """

  def __init__(self, layer0_params, layer0_reduce, layer1_params, layer1_reduce, layer2_params, layer2_reduce,
    rnn_params, layer3_params, fc_params):
    """
    Initialization for the 2dRCNN model. layeri_params, layeri_reduce, fc_params have the same structure of the previous CNN models. 
    rnn_params contains the parameters for the RNN layer: 

      rnn_params = [num_features, hidden_features, num_rnn_layers] 

    """

    super().__init__()

    self.layer0_params_all = []
    self.layer1_params_all = []
    self.layer2_params_all = []
    self.layer3_params_all = layer3_params
    self.rnn_params = rnn_params
    self.fc_all = fc_params 

    # load volume-preserving layers 
    for i in range(5):
      self.layer0_params_all.append(layer0_params[i])
      self.layer1_params_all.append(layer1_params[i])
      self.layer2_params_all.append(layer2_params[i])

    # load reduction layers 
    for j in range(5):
      self.layer0_params_all.append(layer0_reduce[j])
      self.layer1_params_all.append(layer1_reduce[j])
      self.layer2_params_all.append(layer2_reduce[j])
    
    # CNN parameters 
    # indexing: cnn_[calorimeter index]_[conv layer number]
    self.cnn_0_1 = nn.Conv2d(1, self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4])
    self.cnn_0_2 = nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4])
    self.cnn_0_3 = nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[0], (self.layer0_params_all[1], self.layer0_params_all[2]), stride=self.layer0_params_all[3], padding=self.layer0_params_all[4])
    self.cnn_0_4 = nn.Conv2d(self.layer0_params_all[0], self.layer0_params_all[5], (self.layer0_params_all[6], self.layer0_params_all[7]), stride=self.layer0_params_all[8], padding=self.layer0_params_all[9])

    self.cnn_1_1 = nn.Conv2d(1, self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4])
    self.cnn_1_2 = nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4])
    self.cnn_1_3 = nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[0], (self.layer1_params_all[1], self.layer1_params_all[2]), stride=self.layer1_params_all[3], padding=self.layer1_params_all[4])
    self.cnn_1_4 = nn.Conv2d(self.layer1_params_all[0], self.layer1_params_all[5], (self.layer1_params_all[6], self.layer1_params_all[7]), stride=self.layer1_params_all[8], padding=self.layer1_params_all[9])

    self.cnn_2_1 = nn.Conv2d(1, self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4])
    self.cnn_2_2 = nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4])
    self.cnn_2_3 = nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[0], (self.layer2_params_all[1], self.layer2_params_all[2]), stride=self.layer2_params_all[3], padding=self.layer2_params_all[4])
    self.cnn_2_4 = nn.Conv2d(self.layer2_params_all[0], self.layer2_params_all[5], (self.layer2_params_all[6], self.layer2_params_all[7]), stride=self.layer2_params_all[8], padding=self.layer2_params_all[9])
    
    """
    self.num_rnn_layers = rnn_params[2]
    self.cnn_3_1 = nn.Conv1d(self.num_rnn_layers, self.layer3_params_all[0], self.layer3_params_all[1], stride=self.layer3_params_all[3], padding=self.layer3_params_all[4])
    self.cnn_3_2 = nn.Conv1d(self.layer3_params_all[0], self.layer3_params_all[0], self.layer3_params_all[1], stride=self.layer3_params_all[3], padding=self.layer3_params_all[4]) 
    self.cnn_3_2 = nn.Conv1d(self.layer3_params_all[0], self.layer3_params_all[0], self.layer3_params_all[1], stride=self.layer3_params_all[3], padding=self.layer3_params_all[4]) 
    self.cnn_3_3 = nn.Conv1d(self.layer3_params_all[0], self.layer3_params_all[0], self.layer3_params_all[1], stride=self.layer3_params_all[3], padding=self.layer3_params_all[4]) 
    """

    # batchnorm parameters, convolutional layers 
    self.batch_0_1 = nn.BatchNorm2d(self.layer0_params_all[0], track_running_stats=False)
    self.batch_0_2 = nn.BatchNorm2d(self.layer0_params_all[5], track_running_stats=False)

    self.batch_1_1 = nn.BatchNorm2d(self.layer1_params_all[0], track_running_stats=False)
    self.batch_1_2 = nn.BatchNorm2d(self.layer1_params_all[5], track_running_stats=False)

    self.batch_2_1 = nn.BatchNorm2d(self.layer2_params_all[0], track_running_stats=False)
    self.batch_2_2 = nn.BatchNorm2d(self.layer2_params_all[5], track_running_stats=False)

    #self.batch_3_1 = nn.BatchNorm1d(self.layer3_params_all[0], track_running_stats=False)

    # batchnorm parameters, fc layers 
    self.batch_fc_1 = nn.BatchNorm1d(self.fc_all[1], track_running_stats=False)
    self.batch_fc_2 = nn.BatchNorm1d(self.fc_all[2], track_running_stats=False)
    self.batch_fc_3 = nn.BatchNorm1d(self.fc_all[3], track_running_stats=False)

    # RNN parameters 
    self.rnn = nn.RNN(self.rnn_params[0], self.rnn_params[1], self.rnn_params[2])

    # FC parameters 
    self.lin_1 = nn.Linear(self.fc_all[0], self.fc_all[1])
    self.lin_2 = nn.Linear(self.fc_all[1], self.fc_all[2])
    self.lin_3 = nn.Linear(self.fc_all[2], self.fc_all[3])
    self.lin_final = nn.Linear(self.fc_all[3], self.fc_all[4])
    
    self.modelName = "rnn_model" 

  def forward(self, l0, l1, l2):
    """
    Forward pass: CNN preprocess --> RNN --> CNN process --> fc --> scores 
    """

    # CNN forward pass for the 0th calorimeter layer image (input: l0)
    cnn_0 = self.cnn_0_1(l0) 
    cnn_0 = self.batch_0_1(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_2(cnn_0) 
    cnn_0 = self.batch_0_1(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_3(cnn_0)
    cnn_0 = self.batch_0_1(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_4(cnn_0)
    cnn_0 = self.batch_0_2(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)

    # CNN forward pass for the 1st calorimeter layer image (input: l1)
    cnn_1 = self.cnn_1_1(l1) 
    cnn_1 = self.batch_1_1(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_2(cnn_1) 
    cnn_1 = self.batch_1_1(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_3(cnn_1)
    cnn_1 = self.batch_1_1(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_4(cnn_1)
    cnn_1 = self.batch_1_2(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)

    # CNN forward pass for the 2nd calorimeter layer image (input: l2)
    cnn_2 = self.cnn_2_1(l2) 
    cnn_2 = self.batch_2_1(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_2(cnn_2) 
    cnn_2 = self.batch_2_1(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_3(cnn_2)
    cnn_2 = self.batch_2_1(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_4(cnn_2)
    cnn_2 = self.batch_2_2(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)

    # rnn layer, with some preprocessing first 
    rnn_input = torch.cat((cnn_0, cnn_1, cnn_2), dim=1)
    rnn_input = rnn_input.permute(1, 0, 2, 3) # takes (batch_size, 3, 3, 6) --> (3, batch_size, 3, 6)
    batch_size = list(cnn_0.size())[0] 
    rnn_input = rnn_input.view(3, batch_size, -1) # reshape to (3, batch_size, 3*6) 
    rnn_out = self.rnn(rnn_input)
    rnn_final = rnn_out[1] # extract final hidden state 

    # CNN processing layer 
    # currently, these are volume-preserving layers 
    """
    rnn_final = rnn_final.permute(1, 0, 2)  # takes (num_layers, batch_size, hidden_size) --> (batch_size, num_layers, hidden_size)
    cnn_3 = self.cnn_3_1(rnn_final) 
    cnn_3 = self.batch_3_1(cnn_3) 
    cnn_3 = nn.ReLU()(cnn_3)
    cnn_3 = self.cnn_3_2(cnn_3)
    cnn_3 = self.batch_3_1(cnn_3)
    cnn_3 = nn.ReLU()(cnn_3)
    cnn_3 = self.cnn_3_3(cnn_3)
    cnn_3 = self.batch_3_1(cnn_3)
    cnn_3 = nn.ReLU()(cnn_3)
    """

    # fc layer preprocessing
    #x = flatten(cnn_3)
    #x = flatten(rnn_final) 
    x = rnn_final.view(batch_size, -1) # reshape to (batch_size, 3*6) 

    # fully connected net forward pass 
    fc = self.lin_1(x) 
    fc = self.batch_fc_1(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    fc = self.lin_2(fc) 
    fc = self.batch_fc_2(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    fc = self.lin_3(fc) 
    fc = self.batch_fc_3(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    scores = self.lin_final(fc) 
  
    return scores 

