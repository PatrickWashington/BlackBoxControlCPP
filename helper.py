import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
import pickle
import time
import csv
# from copy import copy, deepcopy

import scipy
import scipy.linalg

def train_nn(inputs,outputs,activate='sigmoid',BATCH_SIZE=1000,EPOCH=75,filename='trainednetwork'):
    inputsize = inputs.size()[1]
    print(inputsize)
    outputsize = outputs.size()[1]
    numsample = inputs.size()[0]

    numtrain = int(numsample * 0.9)
    numval = int(numsample * 0.1)

    xtrain = inputs[:,:numtrain]
    ytrain = outputs[:,:numtrain]
    xval = inputs[:,-numval:]
    yval = outputs[:,-numval:]
    # xtrain = inputs
    # ytrain = outputs
    # xval = inputs
    # yval = outputs

    hiddensize = 10*inputsize

    if activate == 'sigmoid':
        activation = torch.nn.Sigmoid()
    elif activate == 'tanh':
        activation = torch.nn.Tanh()
    elif activate == 'relu':
        activation = torch.nn.ReLU()

    # activation = torch.nn.Tanh()
    # activation = torch.nn.Sigmoid()
    # activation = torch.nn.ReLU()
    # activation = torch.nn.LeakyReLU()

    net = torch.nn.Sequential(
        torch.nn.Linear(inputsize, hiddensize),
        activation,
        torch.nn.Linear(hiddensize, hiddensize),
        activation,
        torch.nn.Linear(hiddensize, hiddensize),
        activation,
        torch.nn.Linear(hiddensize, hiddensize),
        activation,
        torch.nn.Linear(hiddensize, hiddensize),
        activation,
        torch.nn.Linear(hiddensize, outputsize)
    )

    # optimizer = torch.optim.RMSprop(net.parameters())
    # optimizer = torch.optim.AdamW(net.parameters())
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.5)
    loss_func = torch.nn.MSELoss()

    torch_dataset = Data.TensorDataset(xtrain, ytrain)
    
    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=1,)

    MSE_history = []

    for epoch in range(EPOCH):
        print("Epoch " + str(epoch+1) + " of " + str(EPOCH))
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            prediction = net(batch_x)     # input x and predict based on x
            loss = loss_func(prediction, batch_y)     # must be (1. nn output, 2. target)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        MSE_history.append(validate_nn(net,xval,yval))

        
        # if epoch%100 == 0:
        #     validation_set = torch.Tensor(generate_true_data(int(numsample/100)))

        # if epoch%1 == 0:
        #     MSE_history[epoch] = validate_nn(net,validation_set)
        #     print(MSE_history[epoch])
    
    plt.figure()
    plt.plot(MSE_history)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.show()

    torch.save(net,filename+'.pth')
    print('Saved network (pytorch) to',filename+'.pth')

    nn2csv(net,filename+'.csv',activate)
    print('Saved network (csv) to',filename+'.csv')

    return net

def validate_nn(net,xval,yval):
    y_true = yval.detach().numpy()
    y_net = net(xval).detach().numpy()
    return np.mean((y_net - y_true)**2)

def nn2csv(net,filename,activate):
    kk = 0
    for layer in net.parameters():
        if kk == 0:
            insize = layer.size()[1]
            hiddensize = layer.size()[0]
        if layer.size()[0] != hiddensize:
            outsize = layer.size()[0]
        kk += 1
    numhidden = int(kk/2 - 2)

    kk = 0
    with open(filename,mode='w') as f:
        w = csv.writer(f,delimiter=',')
        w.writerow([insize,hiddensize,outsize,numhidden,activate])
        for layer in net.parameters():
            for ii in range(layer.size()[0]):
                if kk%2 == 0:
                    w.writerow(layer[ii,:].tolist())
                else:
                    w.writerow([layer[ii].tolist()])
            kk += 1
        