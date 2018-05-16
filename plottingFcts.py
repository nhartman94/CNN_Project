'''
Functions to make some of the basic plots that we might be looking at
(1) Train / val loss and acc
(2) Discriminant plots
(3) Bkg rej vs. sig eff roc curves
(4) Perhaps sig eff as a function of the p'cle energy 
(5) Perhaps bkg rej as a function of the p'cle energy 
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

import torch
from torch.nn import Softmax

# Some useful global variables to use across functions
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype = torch.float32


def trainingMetrics(hist, modelName):

    '''
    Plot the training and val training metrics for a model. 
    
    Input: 
    - hist: A dictionary w/ keys loss, acc, val_acc
    - modelName: String for saving the figures 
    '''

    # Get the training loss / acc cuves
    loss     = hist['loss']
    acc      = hist['acc']
    val_acc  = hist['val_acc']

    iters = np.arange(1,loss.size+1)
    epochs = np.arange(1,acc.size+1)

    plt.plot(epochs,loss,label='training')
    #plt.plot(epochs,val_loss,label='validation')
    plt.xlabel('iterations',fontsize=14)
    plt.ylabel('cross-entropy loss',fontsize=14)
    plt.legend()
    plt.savefig('../figures/loss_{}.jpg'.format(modelName))

    plt.figure()
    plt.plot(epochs,acc,label='training')
    plt.plot(epochs,val_acc,label='validation')
    plt.xlabel('epochs',fontsize=14)
    plt.ylabel('accuracy',fontsize=14)
    plt.ylim(0.775,.85)
    plt.legend()
    plt.savefig('../figures/acc_{}.jpg'.format(modelName))
    plt.show()


def plotProbs(m, loader):

    '''
    Plot histograms of the output nodes        
      0: gamma
      1: pi+
      2: e+

   Input: 
        m: A PyTorch model
        loader: DataLoader 
    '''

    predictions = []

    with torch.no_grad():
        for l0, l1, l2, y in loader:
            l0 = l0.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            l1 = l1.to(device=device, dtype=dtype)
            l2 = l2.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = m(l0, l1, l2)
            probs = Softmax(dim=1)(scores)
            #print(probs.shape)
            predictions.append(probs.numpy())


    # Need stack the probabilities from the mini-batches
    predictions = np.concatenate(tuple(predictions),axis=0)

    '''
    To do: Plot the output for the nodes for each type of shower 
    '''
 

def sigBkgEff(m, loader, signalNode):
    
    '''
        Given a model, make the histograms of the model outputs to get the ROC curves.
        
        Input: 
            m: A PyTorch model
	    loader: DataLoader 
            signalNode: Which class you choose to be the "signal"       
 		0: gamma
		1: pi+
                2: e+

        Output:
            effs: A list with 3 entries for the l, c, and b effs
    '''
   
    predictions = []
    y_test = []

    with torch.no_grad():
        for l0, l1, l2, y in loader:
            l0 = l0.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            l1 = l1.to(device=device, dtype=dtype)
            l2 = l2.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = m(l0, l1, l2)
            probs = Softmax(dim=1)(scores)
            #print(probs.shape)
            predictions.append(probs.numpy())
            y_test.append(y.numpy())
    
    # Need stack the probabilities from the mini-batches
    predictions = np.concatenate(tuple(predictions),axis=0) 
    y_test = np.concatenate(tuple(y_test),axis=0) 

    if signalNode == 0: 
        disc = np.log(np.divide(predictions[:,0], predictions[:,1] + predictions[:,2]))
        xlabel = '$D_\gamma = \ln [ p_\gamma / (p_\pi + p_e ) ]$'
    elif signalNode == 1: 
        disc = np.log(np.divide(predictions[:,1], predictions[:,0] + predictions[:,2]))
        xlabel = '$D_pi = \ln [ p_\pi / (p_\gamma + p_e ) ]$'
    elif signalNode == 2: 
        disc = np.log(np.divide(predictions[:,2], predictions[:,1] + predictions[:,0]))
        xlabel = '$D_e = \ln [ p_e / (p_\gamma + p_\pi ) ]$'
    else:
        print("Error: {} is not a valid signalNode... reutrning".format(signalNode))
        return

    # To make sure you're in a valid range for the discriminant, set the range
    # using the observed data   
    discMax = np.max(disc)
    discMin = np.min(disc)

    print(disc.shape)   
    print(discMin, discMax)   
 
    myRange=(discMin,discMax) 
    nBins = 200
        
    effs = []
    for output, flavor in zip([0,1,2],['$\gamma$','$\pi^+$','$e^+$']):
    
        ix = (y_test == output)

        # Plot the discriminant output
        nEntries, edges , _ = plt.hist(disc[ix],alpha=0.5,label='{} shower'.format(flavor),
                                      bins=nBins, range=myRange, normed=True, log=True)

        # Calculate the baseline signal and bkg efficiencies 
        eff = np.add.accumulate(nEntries[::-1]) / np.sum(nEntries)
        effs.append(eff)
    
    plt.legend()
    plt.xlabel(xlabel,fontsize=14)
    plt.ylabel('"Normalized" counts')
    #plt.savefig('../figures/disc_{}.jpg'.format(m.modelName))
    plt.show()

    return effs

def plotROC(teffs, beffs, labels, title='', tag='', styles=None, colors=None, ymax=-1):
    '''
    Plot the ROC curves for a list of experiments you've run
    
    Inputs:
        teffs: List for the signal efficiencies
        beffs: List for the background efficiencies
        labels: List of identifiers for the experiment for the legend
        title: Title for the plot
        tag: An option for the tag that you could append to the filename to save the plot
             The deault option of an empty tag won't produce any plots
        styles: If given, should be list of the same length as teff and beffs for the 
                styles of the plots
        colors: If given, should be a list of colors to use for the plots
    
    Note: This function expects the list arguments to all be the same length, but because
    of the way python's zip argument handles varying sized lists, if they aren't the same
    length it will zip to the shortest list.
    
    '''

    if styles is None:
        styles = ['-' for i in teffs]
    
    if colors is None:
        colors = ['C{}'.format(i) for i in range(len(teffs))]
   
    plt.figure()    
    for teff, beff, label, style, color in zip(teffs, beffs, labels, styles, colors):

        plt.semilogy(teff, np.divide(1,beff), style, color=color, label=label)
        
    plt.xlabel('Signal Efficiency',fontsize=14)
    plt.ylabel('Background Rejection',fontsize=14)
    if len(title) > 0:
        plt.title(title)
        
    # Set the axes to be in a reasonable range    
    plt.xlim(0.6,1)
    plt.ylim(1,1e3)
    plt.legend(loc='best')

    if len(tag) != 0:
        plt.savefig('../figures/roc_{}.jpg'.format(tag))

    plt.show()


