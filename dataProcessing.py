'''
Return DataLoaders to iterate over the Pytorch data
'''
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# There are some global variables that these functions use
trainFrac = .6
valFrac = .1
testFrac = .3

nClasses = 3


class emShowersDatasetFlat(Dataset):
    """EM showers dataset"""
    
    def __init__(self, N, relPath='../data/', trainFrac=trainFrac, transform=None):
        """
        
        Instantiates a class which then returns examples as a tuple for the 
        image labels, and the truth labels are:
            0 (gamma), 1 (pi-plus), 2 (positron)
        
        Args:
            N: The number of images we have for each particle class
            relPath: The relative path to where the hdf5 files live
        
        Caveat: I just manually subtracted the mean images from the training
        set and did the tensor transforms, but actually, this would be done
        a lot more elegantly with Pytorch's transforms
        """
        
        d_gamma  = h5py.File(relPath+'gamma.hdf5', 'r')
        d_piplus = h5py.File(relPath+'piplus.hdf5', 'r')
        d_eplus  = h5py.File(relPath+'eplus.hdf5', 'r')
        
        # Subtract the mean image from the training set in each of the layers
        l0_gamma_mean = np.mean(d_gamma['layer_0'][:int(trainFrac*N)],axis=0)
        l1_gamma_mean = np.mean(d_gamma['layer_1'][:int(trainFrac*N)],axis=0)
        l2_gamma_mean = np.mean(d_gamma['layer_2'][:int(trainFrac*N)],axis=0)
        
        l0_piplus_mean = np.mean(d_piplus['layer_0'][:int(trainFrac*N)],axis=0)
        l1_piplus_mean = np.mean(d_piplus['layer_1'][:int(trainFrac*N)],axis=0)
        l2_piplus_mean = np.mean(d_piplus['layer_2'][:int(trainFrac*N)],axis=0)
        
        l0_eplus_mean = np.mean(d_eplus['layer_0'][:int(trainFrac*N)],axis=0)
        l1_eplus_mean = np.mean(d_eplus['layer_1'][:int(trainFrac*N)],axis=0)
        l2_eplus_mean = np.mean(d_eplus['layer_2'][:int(trainFrac*N)],axis=0)

        layer0_mean = (l0_gamma_mean + l0_piplus_mean + l0_eplus_mean) / 3.
        layer1_mean = (l1_gamma_mean + l1_piplus_mean + l1_eplus_mean) / 3.
        layer2_mean = (l2_gamma_mean + l2_piplus_mean + l2_eplus_mean) / 3.
        
        layer0 = np.vstack((d_gamma['layer_0'][:], d_piplus['layer_0'][:], d_eplus['layer_0'][:]))
        layer1 = np.vstack((d_gamma['layer_1'][:], d_piplus['layer_1'][:], d_eplus['layer_1'][:]))
        layer2 = np.vstack((d_gamma['layer_2'][:], d_piplus['layer_2'][:], d_eplus['layer_2'][:]))
        
        # Reshape the tensors as NxCxHxW, with C=1 in this case :)
        m0,h0,w0 = layer0.shape
        layer0 = layer0.reshape(m0,1,h0,w0)

        m1,h1,w1 = layer1.shape
        layer1 = layer1.reshape(m1,1,h1,w1)

        m2,h2,w2 = layer2.shape
        layer2 = layer2.reshape(m2,1,h2,w2)

        # Test to make sure that all of the datasets are the same length
        self.layer0 = torch.from_numpy(layer0 - layer0_mean).type(torch.FloatTensor) 
        self.layer1 = torch.from_numpy(layer1 - layer1_mean).type(torch.FloatTensor)
        self.layer2 = torch.from_numpy(layer2 - layer2_mean).type(torch.FloatTensor)

  
        # Get the y labels
        self.y = torch.from_numpy(np.concatenate((np.zeros(N), np.ones(N), 2*np.ones(N))))
        
    def __len__(self):
        return self.layer0.shape[0] 

    def __getitem__(self, idx):
        
        return self.layer0[idx], self.layer1[idx], self.layer2[idx], self.y[idx]

def getDataLoaders(batch_size=64, N=100000):
    '''

    Input: 
        batch_size
        N: Number of events / particle, 100k uses all the available data
 
    Returns: loader_train, loader_val, loader_test
        DataLoaders for the train, val, and test sets
 
    '''

    nClasses = 3
    dset = emShowersDatasetFlat(N=N)
    
    idxTrain = []
    idxVal = []
    idxTest = []
    
    for i in range(nClasses):
        
        idxTrain += [j for j in range(i*N, int((i+trainFrac)*N))]
        idxVal += [j for j in range(int((i+trainFrac)*N), int((i+trainFrac+valFrac)*N))]
        idxTest += [j for j in range(int((i+trainFrac+valFrac)*N), (i+1)*N)]
    
    loader_train = DataLoader(dset, batch_size=batch_size, sampler=SubsetRandomSampler(idxTrain))
    loader_val = DataLoader(dset, batch_size=batch_size, sampler=SubsetRandomSampler(idxVal))
    loader_test = DataLoader(dset, batch_size=batch_size, sampler=SubsetRandomSampler(idxTest))

    return loader_train, loader_val, loader_test
