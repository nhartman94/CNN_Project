import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from train import check_accuracy, train 
from dataProcessing import getDataLoaders
from models import FCNet, CNN_3d, rnn_2dCNN

"""
Function to plot the maximum validation accuracy achieved by a model, as a function of the number of images 
the model was trained on. 
Parameters: 
    - num_images = [1e1, 1e2, ..., 1e5] --> array holding number of images to train each model on, up to the total 1e5
    - params_all = [layer0_params, layer0_reduce, ..., rnn_params, fc_params] --> array holding the parameters for each layer 
    - learn_rate: model's learning rate 
    - loader_train, loader_val: train/test data sets 
    - epochs: number of epochs to train all models for 
    - batch_size: self-explanatory 
Plots number of images vs. best validation acc achieved by model, and returns all the models trained 
""" 

def num_valAcc(num_images, params_all, learn_rate, epochs, batch_size):

    # dictionary to store best models obtained during training  
    all_models = {} 
    # array to store best validation accuracy 
    best_val = [] 

    for num in num_images: 

        # load appropriate number of images 
        loader_train, loader_val, __ = getDataLoaders(batch_size=batch_size, N=num)
        
        """
        IMPORTANT NOTE: to train with the 3d CNN model/fc model, replace rnn_2dCNN with CNN_3d/FCNet model below 
        """

        model = rnn_2dCNN(params_all[0], params_all[1], params_all[2], params_all[3], params_all[4], params_all[5], params_all[6], params_all[7], params_all[8])

        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        hist, bestModel = train(loader_train, loader_val, model, optimizer, epochs=epochs, returnBest=True, verbose=False) 

        # add best trained model for this iteration to dictionary 
        # e.g. model1000 --> model trained on 1000 images 
        all_models["model" + str(num)] = bestModel
        # add best validation accuracy to array 
        best_acc = check_accuracy(loader_val, bestModel, returnAcc=True, verbose=False)
        best_val.append(best_acc)
        
        print("Finished training with " + str(num) + " images.") 
        print("Best val accuracy: %f" %best_acc) 
        print("________________________________________________") 
        
    # plot number of images vs best validation accuracy 
    plt.plot(num_images, best_val)
    plt.xlabel("Number of training images", fontsize=14)
    plt.ylabel("Best validation accuracy", fontsize=14)

    return all_models 

