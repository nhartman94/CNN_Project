'''
Some useful functions for training the Pytorch networks stolen from
the PyTorch nb in assignment 2 :-)
'''
import torch
import torch.nn.functional as F
import copy

# Some useful global variables to use across functions
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype = torch.float32

def check_accuracy(loader, model, returnAcc=False, verbose=False):

    '''
    Check the accuracy of the model

    Inputs:
        loader: A DataLoader object, i.e, for the val or test st
        model: A Pytorch model to check the accuracy on
        returnAcc: If true, the function will return the calculated accuracy

    '''
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
        
    if verbose: 
        print(device)
    model = model.to(device=device)

    with torch.no_grad():
        for l0, l1, l2, y in loader:
            l0 = l0.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            l1 = l1.to(device=device, dtype=dtype)
            l2 = l2.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(l0, l1, l2)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        if verbose: 
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

        if returnAcc:
            return acc

def check_loss(loader, model):
    '''
    Calculate the loss
    '''
    with torch.no_grad():
        for l0, l1, l2, y in loader:
            l0 = l0.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            l1 = l1.to(device=device, dtype=dtype)
            l2 = l2.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            # Calculate the loss
            scores = model(l0, l1, l2)
            loss = F.cross_entropy(scores, y) 


def train(loader_train, loader_val, model, optimizer, epochs=1, returnBest=False, verbose=False):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - loader_train: DataLoader for the training set
    - loader_val: DataLoader for the validation set
    - model: A PyTorch model to train.
    - optimizer: An optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: 
    - history: A dictionary object with the training loss over each iteration,
               and training / val accuracies for each epoch
    If returnBest is True, it also returns the model at the best epoch

    """
    print_every=100

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    hist = {}
    hist['loss'] = []
    hist['acc'] = []
    #hist['val_loss'] = []
    hist['val_acc'] = []

    bestModel = None
    bestValAcc = 0

    for e in range(epochs):
        if verbose: 
            print("\nEpoch {}/{}:".format(e+1,epochs))

        for t, (l0, l1, l2, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            l0 = l0.to(device=device, dtype=dtype)
            l1 = l1.to(device=device, dtype=dtype)
            l2 = l2.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(l0, l1, l2)
            loss = F.cross_entropy(scores, y)
            hist['loss'].append(loss.item())

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
            if verbose: 
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    check_accuracy(loader_val, model)
                    print()

        # Save the acc / epoch
        hist['acc'] .append(check_accuracy(loader_train, model, returnAcc=True)) 
        hist['val_acc'] .append(check_accuracy(loader_val, model, returnAcc=True)) 

        # Check if this model has the best validation accuracy 
        if hist['val_acc'][-1] > bestValAcc:
            bestValAcc = hist['val_acc'][-1] 
            bestModel = copy.deepcopy(model)

    # Save the weights for the best model
    if model.modelName is not None:
        # https://pytorch.org/docs/master/notes/serialization.html
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
        torch.save(bestModel.state_dict(), "../models/{}.pt".format(model.modelName))
        torch.save(bestModel.state_dict(), "models/{}.pt".format(model.modelName))

    if returnBest:
        return hist, bestModel
    else:
        return hist


def train_ThreeCNN(loader_train, loader_val, layer0_model, layer1_model, layer2_model, fc_model, optimizer, epochs=1):
    """
    Train a model on the Micky et al calorimeter dataset. Note that because this training function must concatenate the outputs of 
    three seperate CNNs, it takes four models as input: one model for each calorimeter-layer preprocessing CNN, and a final FC model that 
    outputs class scores. 
    """
    print_every=100

    # move model parameters to CPU/GPU
    layer0_model = layer0_model.to(device=device)  
    layer1_model = layer1_model.to(device=device)
    layer2_model = layer2_model.to(device=device)
    fc_model = fc_model.to(device=device)

    hist = {}
    hist['loss'] = []
    hist['acc'] = []
    #hist['val_loss'] = []
    hist['val_acc'] = []

    for e in range(epochs):

        for t, (l0, l1, l2, y) in enumerate(loader_train):
            # put models to training mode
            layer0_model.train()
            layer1_model.train()
            layer2_model.train()
            fc_model.train()
            
            l0 = l0.to(device=device, dtype=dtype)
            l1 = l1.to(device=device, dtype=dtype)
            l2 = l2.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            #scores = model(l0, l1, l2)

            # compute intermediate tensors from the CNN preprocessing layers 
            tensor_L0 = layer0_model(l0)
            tensor_L1 = layer1_model(l1) 
            tensor_L2 = layer2_model(l2)

            # flatten and concatenate tensors for FC layer 
            x = torch.cat((flatten(tensor_L0),flatten(tensor_L1),flatten(tensor_L2)), dim=1)
            scores = fc_model(x)

            loss = F.cross_entropy(scores, y)
            hist['loss'].append(loss.item)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()

        # Save the acc / epoch
        hist['acc'] .append(check_accuracy(loader_train, model)) 
        hist['val_acc'] .append(check_accuracy(loader_val, model)) 

    return hist
