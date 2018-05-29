# Classifying Electromagnetic Showers from Calorimeter Images with CNNs

## Algorithms

Our initial steps are to flatten the three calorimeter images and concatenate into one long vector, and then feed this vector into a fully-connected network to attempt particle classification. 

After we experiment with the fully connected network, we'll preprocess each image individually with a CNN (e.g. one CNN for each image), choosing pooling and filters appropriately to make each CNN output 3 x 6. These three CNN activation layers will be flattened into a single vector, and provided to a fully-connected layer for classification. 

Alternatively, we could combine the three activation maps from the above CNNs into a single image (i.e. each activiation map represents a color channel). We could then use this image as input to another CNN, that has a finaly fully-connected layer. 

We'll use the following baseline architecture for the CNNs: 

    (1) 3 x ([Conv --> Batchnorm --> ReLU] x N) --> FC --> Softmax 
    (2) 3 x ([Conv --> Batchnorm --> ReLU] x N) --> [Conv --> Batchnorm --> ReLU] x M --> FC --> Softmax
    
In the above, the "3 x" denotes three seperate CNN layers for each input image, after which the resulting activation maps are then (1) flattened together and used as input to the FC network, or (2) stacked into a 3-channel image and used as input to another CNN layer. N and M denote the number of layers in each CNN layer. 

Future work will include processing images with a recurrent CNN and using 3D convolutions to process the input images.  

Authors: 
    Nicole Hartman and Sean Mullane, Stanford University Department of Physics

Most esteemed mentor: 
    Michael Kagan, SLAC/CERN-ATLAS 
    
## Dataset

We will be using the jet images of the showers in the EM calorimeter produced
by Micky, et. al.
https://data.mendeley.com/datasets/pvn3xc3wy5/1

## To Do:
**Visualization:**
- t-SNE for the separation of classes
- Occlusion test to see whch sections of the input img are most important for classifying the output
- 

**Algorithms:**
- 3d-CNN
- RCNN
- Train a GAN, and then just use the encoder
- Train a VAE!
