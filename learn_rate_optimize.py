import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from train import check_accuracy, train 
from dataProcessing import getDataLoaders
from models import rnn_2dCNN

"""
Performs a random search for the learning rate in log space. 
"""

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(device) 

def lr_optimize(num_images, params_all, epochs, batch_size, iterations):
    """
    Inputs; 
        - num_images: number of images to train on 
        - params_all: [layer0_params, layer0_reduce, ..., rnn_params, fc_params] --> array holding the parameters for each layer 
        - epochs, batch_size: self-explanatory 
        - iterations: total number of models to train, total number of random samplings in log space
    Plots the validation accuracy as a function of learning rate for each model considered. Also returns an array holding all the 
    validation accuracies, an array holding the learning rates corresponding to these accuracies, and the model that has the highest validation accuracies across all models considered in the function. 
    """
    # arrays to hold validation accuracy, learning rates 
    val_acc = []
    learn_rates = []
    best_acc = 0.0
    best_model = None
     
    i = 0
    # main loop 
    while i < iterations: 
        #exponent = np.random.uniform(-6.0, -2.0)
        #lr = 10**exponent 
        lr = np.random.uniform(2e-3, 1e-2) 

        loader_train, loader_val, __ = getDataLoaders(batch_size=batch_size, N=num_images)

        model = rnn_2dCNN(params_all[0], params_all[1], params_all[2], params_all[3], params_all[4], params_all[5], params_all[6], params_all[7], params_all[8])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        hist, bestModel = train(loader_train, loader_val, model, optimizer, epochs=epochs, returnBest=True, verbose=False)

        # determine validation acc 
        model_acc = check_accuracy(loader_val, model, returnAcc=True, verbose=False)
        if model_acc > best_acc:
            best_acc = model_acc
            best_model = model 
            print("Assigned best validation accuracy:")

        val_acc.append(model_acc)
        learn_rates.append(lr)

        i += 1 
        print(model_acc, lr)
        print("Finished iteration %f" %i)
        print("__________________________")
        
    return val_acc, learn_rates, best_model  

def lr_filters_optimize(filter_range, num_images, params_all, epochs, batch_size, iterations):
    """
    Optimizes both the learning rate and number of filters in the volume-preserving layers of the three CNN preprocessors. 
    Uses a random search in log space for the lr, and a random search between filter_range[0] and filter_range[1] for the 
    number of filters. 
    Inputs: same as above, except for filter_range = [range_min, range_max] giving the range of filter numbers over which to choose 
    """
    # arrays to hold validation accuracy, learning rates, filter numbers
    val_acc = []
    learn_rates = []
    num_filters = []
    best_acc = 0.0
    best_model = None
     
    i = 0
    # main loop 
    while i < iterations: 
        exponent = np.random.uniform(-6, -2)
        lr = 10**exponent 
        filter_num = int(np.random.uniform(filter_range[0], filter_range[1]))

        # load selected filter number 
        params_all[0][0] = filter_num
        params_all[2][0] = filter_num
        params_all[4][0] = filter_num 

        loader_train, loader_val, __ = getDataLoaders(batch_size=batch_size, N=num_images)
        
        model = rnn_2dCNN(params_all[0], params_all[1], params_all[2], params_all[3], params_all[4], params_all[5], params_all[6], params_all[7], params_all[8])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        hist, bestModel = train(loader_train, loader_val, model, optimizer, epochs=epochs, returnBest=True, verbose=False)

        # determine validation acc 
        model_acc = check_accuracy(loader_val, model, returnAcc=True, verbose=False)
        if model_acc > best_acc:
            best_acc = model_acc
            best_model = model 
            print("Assigned best validation accuracy:")

        val_acc.append(model_acc)
        learn_rates.append(lr)
        num_filters.append(filter_num)

        i += 1 
        print(model_acc, lr)
        print(filter_num) 
        print("Finished iteration %f" %i)
        print("__________________________")

    return val_acc, learn_rates, num_filters, best_model 

def filters_optimize(filter_range, learn_rate, num_images, params_all, epochs, batch_size, iterations):
    """
    Almost the same as above, but optimizes on filters with a single learning rate 
    """
    # arrays to hold validation accuracy, learning rates, filter numbers
    val_acc = []
    num_filters = []
    best_acc = 0.0
    best_model = None
     
    i = 0
    # main loop 
    while i < iterations: 
        filter_num = int(np.random.uniform(filter_range[0], filter_range[1]))
        print("Selected filter number: %f" %filter_num)

        # load selected filter number 
        params_all[0][0] = filter_num
        params_all[2][0] = filter_num
        params_all[4][0] = filter_num 

        loader_train, loader_val, __ = getDataLoaders(batch_size=batch_size, N=num_images)

        model = rnn_2dCNN(params_all[0], params_all[1], params_all[2], params_all[3], params_all[4], params_all[5], params_all[6], params_all[7], params_all[8])
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        hist, bestModel = train(loader_train, loader_val, model, optimizer, epochs=epochs, returnBest=True, verbose=False)

        # determine validation acc 
        model_acc = check_accuracy(loader_val, model, returnAcc=True, verbose=False)
        if model_acc > best_acc:
            best_acc = model_acc
            best_model = model 
            print("Assigned best validation accuracy:")

        val_acc.append(model_acc)
        num_filters.append(filter_num)

        i += 1 
        print(model_acc, filter_num)
        print("Finished iteration %f" %i)
        print("__________________________")

    return val_acc, num_filters, best_model 
