import numpy as np
import torch
from helper import train_nn
import os
import csv
import matplotlib.pyplot as plt

os.system('make release')

sysname = 'block'
task = 'gendata'
numsample = 10000
filestart = 'block/block'
cmd = (' ').join(['./run',sysname,task,str(numsample),filestart])
os.system(cmd)

filename = 'block/block_trainingdata.csv'
# filename = 'block/block_ilqr_result.csv'

def loaddata(filename):
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
    
    state = torch.Tensor(data[:2,:-1].T)
    control = torch.Tensor(data[2:3,:-1].T)
    # nextstate = torch.Tensor(data[3:5,:].T)
    nextstate = torch.Tensor(data[:2,1:].T)

    return state,control,nextstate

# s1,c1,n1 = loaddata('block/block_ilqr_result1.csv')
# s2,c2,n2 = loaddata('block/block_ilqr_result2.csv')
# s3,c3,n3 = loaddata('block/block_ilqr_result3.csv')
# s4,c4,n4 = loaddata('block/block_ilqr_result4.csv')
# # s5,c5,n5 = loaddata('block/block_ilqr_result5.csv')
# state = torch.cat((s1,s2,s3,s4),dim=0)
# control = torch.cat((c1,c2,c3,c4),dim=0)
# nextstate = torch.cat((n1,n2,n3,n4),dim=0)
state,control,nextstate = loaddata(filename)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data[0,:],data[1,:],data[2,:],marker='.')
# plt.show()

a = 'tanh'
e = 100
b = 100

print('Starting first network')
train_nn(torch.cat((state,control),dim=1),nextstate,activate=a,EPOCH=e,BATCH_SIZE=b,filename=filestart+'_statecontrol2next')
print('Starting second network')
train_nn(torch.cat((control,nextstate),dim=1),state,activate=a,EPOCH=e,BATCH_SIZE=b,filename=filestart+'_controlnext2state')
print('Starting third network')
train_nn(torch.cat((nextstate,state),dim=1),control,activate='tanh',EPOCH=100,BATCH_SIZE=b,filename=filestart+'_nextstate2control')
