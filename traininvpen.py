import numpy as np
import torch
from helper import train_nn
import os
import csv

sysname = 'invpen'
task = 'gendata'
numsample = 100000
filestart = 'invpen/invpen'
cmd = (' ').join(['./run',sysname,task,str(numsample),filestart])
os.system(cmd)

filename = 'invpen/invpen_indata.csv'
with open(filename) as file:
    filereader = csv.reader(file,delimiter=',')
    ii = 0
    for row in filereader:
        if ii == 0:
            numrow = int(row[0])
            numcol = int(row[1])
            inputs = np.zeros((numrow,numcol))
        else:
            for jj in range(numcol):
                inputs[ii-1,jj] = float(row[jj])
        ii += 1

filename = 'invpen/invpen_outdata.csv'
with open(filename) as file:
    filereader = csv.reader(file,delimiter=',')
    ii = 0
    for row in filereader:
        if ii == 0:
            numrow = int(row[0])
            numcol = int(row[1])
            outputs = np.zeros((numrow,numcol))
        else:
            for jj in range(numcol):
                outputs[ii-1,jj] = float(row[jj])
        ii += 1



inputs = torch.Tensor(inputs.T)
outputs = torch.Tensor(outputs.T)

train_nn(inputs,outputs,activate='tanh',filename=filestart+'_network.csv')
