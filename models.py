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

       self.modelName = "fc_{}_{}_{}_dpt_{}".format(h1_dim, h2_dim, h3_dim,p)

     
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
    
  def __init__(self, layer0_params, layer0_reduce, layer1_params, layer1_reduce, layer2_params, layer2_reduce, fc_params):

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

    # FC parameters 
    self.lin_1 = nn.Linear(self.fc_all[0], self.fc_all[1])
    self.lin_2 = nn.Linear(self.fc_all[1], self.fc_all[2])
    self.lin_3 = nn.Linear(self.fc_all[2], self.fc_all[3])
    self.lin_final = nn.Linear(self.fc_all[3], self.fc_all[4])

  def forward(self, l0, l1, l2):

    """
    Forward pass for the network. Unlike in ThreeCNN, we use here the module API. 
    """

    # CNN forward pass for the 0th calorimeter layer image (input: l0)
    cnn_0 = self.cnn_0_1(l0) 
    cnn_0 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_2(cnn_0) 
    cnn_0 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_3(cnn_0)
    cnn_0 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)
    cnn_0 = self.cnn_0_4(cnn_0)
    cnn_0 = nn.BatchNorm2d(self.layer0_params_all[5])(cnn_0)
    cnn_0 = nn.ReLU()(cnn_0)

    # CNN forward pass for the 1st calorimeter layer image (input: l1)
    cnn_1 = self.cnn_1_1(l1) 
    cnn_1 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_2(cnn_1) 
    cnn_1 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_3(cnn_1)
    cnn_1 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)
    cnn_1 = self.cnn_1_4(cnn_1)
    cnn_1 = nn.BatchNorm2d(self.layer0_params_all[5])(cnn_1)
    cnn_1 = nn.ReLU()(cnn_1)

    # CNN forward pass for the 2nd calorimeter layer image (input: l2)
    cnn_2 = self.cnn_2_1(l2) 
    cnn_2 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_2(cnn_2) 
    cnn_2 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_3(cnn_2)
    cnn_2 = nn.BatchNorm2d(self.layer0_params_all[0])(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)
    cnn_2 = self.cnn_2_4(cnn_2)
    cnn_2 = nn.BatchNorm2d(self.layer0_params_all[5])(cnn_2)
    cnn_2 = nn.ReLU()(cnn_2)

    # flatten, concatenate outputs from CNN forward passes 
    x = torch.cat((flatten(cnn_0),flatten(cnn_1),flatten(cnn_2)), dim=1)

    # fully connected net forward pass 
    fc = self.lin_1(x) 
    fc = nn.BatchNorm1d(self.fc_all[1])(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    fc = self.lin_2(fc) 
    fc = nn.BatchNorm1d(self.fc_all[2])(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    fc = self.lin_3(fc) 
    fc = nn.BatchNorm1d(self.fc_all[3])(fc) 
    fc = nn.ReLU()(fc) 
    fc = nn.Dropout(self.fc_all[5])(fc) 
    scores = self.lin_final(fc) 
  
    return scores

'''
These two sequential models cast the layers as 12x12 images
'''
layer0_12x12 = nn.Sequential(  nn.Conv2d(1,1, (1,8), stride=(1,8)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1,1, (4,1), stride=(4,1)),
                                    nn.ReLU()
                                 )
     
layer2_12x12 = nn.Sequential(nn.ConvTranspose2d(1,1, (1,2), stride=(1,2)),
                                  nn.ReLU())


class CNN_3d(nn.Module):

    '''
    This first transforms the model input layers into a 12x12 dim images, and then
    applies a 3d convolution to the inputs  
    '''

    def __init__(self, nFilters_1 = 16, filter_1=2, stride_1=2, padding_1=1,
                       nFilters_2 = 8,  filter_2=3, stride_2=2, padding_2=1,
                       nLSTM=25, fc_dim=25):
       '''
      
       Inputs:

 
       '''

       super().__init__()

       self.layer0_12x12 = layer0_12x12
       self.layer2_12x12 = layer2_12x12

       nOut = 3
       spatialDim = 12

       self.cnn3d_1 = nn.Conv3d(1, nFilters_1, filter_1, stride_1, padding_1)
       self.bn3d_1 = nn.BatchNorm3d(nFilters_1)

       # Calculate the number of input dimensions seen by each of the inputs
       d1_out = (3 - filter_1[0] + 2*padding_1[0]) / stride_1[0] + 1 
       h1_out = (spatialDim - filter_1[1] + 2*padding_1[1]) / stride_1[1] + 1
       w1_out = (spatialDim - filter_1[2] + 2*padding_1[2]) / stride_1[2] + 1

       print("Output size after the first conv: {},{},{},{}".format(nFilters_1, d1_out, h1_out, w1_out))

       self.cnn3d_2 = nn.Conv3d(nFilters_1, nFilters_2, filter_2, stride_2, padding_2)
       self.bn3d_2 = nn.BatchNorm3d(nFilters_2)

       d2_out = (d1_out - filter_2[0] + 2*padding_2[0]) / stride_2[0] + 1 
       h2_out = (h1_out - filter_2[1] + 2*padding_2[1]) / stride_2[1] + 1
       w2_out = (w1_out - filter_2[2] + 2*padding_2[2]) / stride_2[2] + 1
       print("Output size after the second conv: {},{},{},{}".format(nFilters_2, d2_out, h2_out, w2_out))

       # After the 3d convolutions, flatten and classify the output
       fc_inpt = nFilters_2 * d2_out * h2_out**2 

       self.fc1 = nn.Linear(fc_inpt, h1_dim) 
       self.fc2 = nn.Linear(h1_dim, h2_dim) 
       self.fc3 = nn.Linear(h2_dim, nOut) 

       self.bn1 = nn.BatchNorm1d(h1_dim)
       self.bn2 = nn.BatchNorm1d(h2_dim)
 
       self.dropout = p

       self.modelName = "cnn3d_12x12_C{}_F{}{}{}_S{}{}{}_P{}{}{}_C{}_F{}{}{}_S{}{}{}_P{}{}{}_fc_{}_{}_dpt_{}".format(nFilters_1,*filter_1,*stride_1,*padding_1,\
                        nFilters_2,*filter_2,*stride_2,*padding_2,h1_dim,h2_dim,p)

    def forward(self, layer0, layer1, layer2):

        # Call the functions above to make the input dim of the three layers the same 
        l0 = self.layer0_12x12(layer0).view(-1,1,1,12,12)
        l1 = layer1.view(-1,1,1,12,12)
        l2 = layer2_12x12(layer2).view(-1,1,1,12,12)

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


class CNN2d_LSTM(nn.Module):

    '''
    This first transforms the model input layers into a 12x12 dim images, and then
    applies a 2d convolution to the inputs, and lastly feeds them as inputs to an LSTM 
    '''

    def __init__(self, nFilters_1 = 16, filter_1=(3,4,4), stride_1=(2,2,2), padding_1=(1,1,1),
                       nFilters_2 = 8,  filter_2=(2,2,2), stride_2=(1,2,2), padding_2=(0,1,1),
                       h1_dim=50, h2_dim=25, p=0.5):
       '''
      
       Inputs:

 
       '''

       super().__init__()

       self.layer0_12x12 = layer0_12x12
       self.layer2_12x12 = layer2_12x12

       nOut = 3
       spatialDim = 12

       self.cnn2d_1 = nn.Conv2d(1, nFilters_1, filter_1, stride_1, padding_1)
       self.bn2d_1 = nn.BatchNorm2d(nFilters_1)

       # Calculate the number of input dimensions seen by each of the inputs
       h1_out = (spatialDim - filter_1[1] + 2*padding_1[1]) / stride_1[1] + 1
       w1_out = (spatialDim - filter_1[2] + 2*padding_1[2]) / stride_1[2] + 1

       print("Output size after the first conv: {},{},{}".format(nFilters_1, h1_out, w1_out))

       self.cnn2d_2 = nn.Conv2d(nFilters_1, nFilters_2, filter_2, stride_2, padding_2)
       self.bn2d_2 = nn.BatchNorm2d(nFilters_2)

       h2_out = (h1_out - filter_2[1] + 2*padding_2[1]) / stride_2[1] + 1
       w2_out = (w1_out - filter_2[2] + 2*padding_2[2]) / stride_2[2] + 1
       print("Output size after the second conv: {},{},{}".format(nFilters_2, h2_out, w2_out))

       # After the 3d convolutions, flatten and classify the output
       fc_inpt = nFilters_2 * h2_out * w2_out 

       # After the 3d convolutions, flatten and classify the output
       fc_inpt = nFilters_2 * d2_out * h2_out**2 

       self.fc1 = nn.Linear(fc_inpt, h1_dim) 
       self.fc2 = nn.Linear(h1_dim, h2_dim) 
       self.fc3 = nn.Linear(h2_dim, nOut) 

       self.bn1 = nn.BatchNorm1d(h1_dim)
       self.bn2 = nn.BatchNorm1d(h2_dim)
 
       self.dropout = p

       self.modelName = "cnn3d_12x12_C{}_F{}{}{}_S{}{}{}_P{}{}{}_C{}_F{}{}{}_S{}{}{}_P{}{}{}_fc_{}_{}_dpt_{}".format(nFilters_1,*filter_1,*stride_1,*padding_1,\
                        nFilters_2,*filter_2,*stride_2,*padding_2,h1_dim,h2_dim,p)

    def forward(self, layer0, layer1, layer2):

        # Call the functions above to make the input dim of the three layers the same 
        l0 = self.layer0_12x12(layer0).view(-1,1,1,12,12)
        l1 = layer1.view(-1,1,1,12,12)
        l2 = layer2_12x12(layer2).view(-1,1,1,12,12)

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



