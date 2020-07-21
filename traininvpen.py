import numpy as np
import torch
from helper import train_nn
import os
import csv
import matplotlib.pyplot as plt

os.system('make release')

sysname = 'invpen'
task = 'gendata'
numsample = 10
filestart = 'invpen/invpen'
cmd = (' ').join(['./run',sysname,task,str(numsample),filestart])
os.system(cmd)

filename = 'invpen/invpen_trainingdata.csv'
with open(filename) as file:
    filereader = csv.reader(file,delimiter=',')
    ii = 0
    for row in filereader:
        if ii == 0:
            numrow = int(row[0])
            numcol = int(row[1])
            data = np.zeros((numrow,numcol))
        else:
            for jj in range(numcol):
                data[ii-1,jj] = float(row[jj])
        ii += 1

# plt.figure()
# plt.plot(data[0,:],data[1,:],'.')
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[0,:],data[1,:],data[2,:],marker='.')
plt.show()


state = torch.Tensor(data[:2,:].T)
control = torch.Tensor(data[2:3,:].T)
nextstate = torch.Tensor(data[3:5,:].T)

a = 'tanh'
e = 50

print('Starting first network')
train_nn(torch.cat((state,control),dim=1),nextstate,activate=a,EPOCH=e,filename=filestart+'_statecontrol2next')
print('Starting second network')
train_nn(torch.cat((control,nextstate),dim=1),state,activate=a,EPOCH=e,filename=filestart+'_controlnext2state')
print('Starting third network')
train_nn(torch.cat((nextstate,state),dim=1),control,activate=a,EPOCH=e,filename=filestart+'_nextstate2control')
