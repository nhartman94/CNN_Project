# Classifying Electromagnetic Showers from Calorimeter Images with CNNs

## Overview 

This repository contains code for constructing various neural networks to classify particles from the Micky et. al. dataset based on energy deposition. The map below provides a brief explanation of each file, and a summary of the notebooks that were important to our project's development. An in-depth discussion of the network architectures can be found in the paper "Classifying Electromagnetic Showers from Calorimeter Images with CNNs", by Nicole Hartman and Sean Mullane. 

## Repository Map 

Main Branch:
    
    -dataProcessing.py --> code to organize dataset into layers for use during training/testing 
    -imageNumber_analysis.py --> functions to test how the total number of training images affects the validation accuracy 
    -learn_rate_optimize.py --> code to optimize the learning rate for a given model 
    -models.py --> central file for all models used in the project 
    -plottingFcts.py --> central file for all plotting functions used in the project 
    -train.py --> training code, largely taken from CS231N homeworks 
    
models folder: contains stored models from various training notebooks 

figures folder: contains saved figures from the model visualization notebooks 

Notebooks folder: central folder for all iPython notebooks used in the project. Includes: 

    -Data-Processsing.ipynb --> initial notebook to test data processing functions
    -Ensembling.ipynb --> an attempt at ensembling/average models to asses overall performance 
    -Final_Models_testSet.ipynb --> notebook that ran the test set on the final 2d RCNN models 
    -Final_Test_Vis.ipynb --> notebook that ran visualization functions on the final 2d RCNN models
    -Visualizations.ipynb --> notebook that ran visualization functions on the final 3d CNN models 
    
3d_models folder: contains notebooks that were used to construct and tune the 3d CNN models 

Baseline_CNN_models: contains notebooks that were used to construct and tune the baseline CNN models 
 
Initial_Notebooks: contains notebooks that were used at the beginning of our project to explore the dataset and construct the first fully connected networks 

Recurrent_CNN_Models: contains notebooks that were used to construct and tune the 2d RCNN models


## Contributors 
Authors: 
    Nicole Hartman and Sean Mullane, Stanford University Department of Physics

Most esteemed mentor: 
    Michael Kagan, SLAC/CERN-ATLAS 
    
## Dataset

We will be using the jet images of the showers in the EM calorimeter produced
by Micky, et. al.
https://data.mendeley.com/datasets/pvn3xc3wy5/1
