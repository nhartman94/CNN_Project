# Classifying Electromagnetic Showers from Calorimeter Images with CNNs

## Dataset

We will be using the jet images of the showers in the EM calorimeter produced
by Micky, et. al.
https://data.mendeley.com/datasets/pvn3xc3wy5/1

## Algorithms

Our initial steps are to flatten the three calorimeter images and concatenate into one long vector, and then feed this vector into a fully-connected network to attempt particle classification. 

After we experiment with the fully connected network, we'll preprocess each image individually with a CNN (e.g. one CNN for each image), choosing pooling and filters appropriately to make each CNN output 3 x 6. These three CNN activation layers will be flattened into a single vector, and provided to a fully-connected layer for classification. 

Alternatively, we could combine the three activation maps from the above CNNs into a single image (i.e. each activiation map represents a color channel). We could then use this image as input to another CNN, that has a finaly fully-connected layer. 

We'll use the following baseline architecture for the CNNs: 
    
    

-> Recurrent convolutional neural networks
-> 3d convolutions
-> beta variational auto encoders

## Repo description
(To fill in later)

Authors: 
    Sean Mullane 
    Nicole Hartman

Most esteemed mentor: 
    Michael Kagan

