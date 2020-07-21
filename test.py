import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# print('hello from test.py')

# os.system('make release')

# cmd = './run invpen ilqr'
# os.system(cmd)

# cmd = './run invpen net invpen/invpen_network.csv test invpen/invpen_ilqr_result.csv'
# os.system(cmd)

# print('I am here now')


# filename = 'invpen/net_invpen_test_result.csv'
# with open(filename) as file:
#     filereader = csv.reader(file,delimiter=',')
#     ii = 0
#     for row in filereader:
#         if ii == 0:
#             numrow = int(row[0])
#             numcol = int(row[1])
#             data = np.zeros((numrow,numcol))
#         else:
#             for jj in range(numcol):
#                 data[ii-1,jj] = float(row[jj])
#         ii += 1

filename = 'invpen/invpen_ilqr_result.csv'
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

# data = data[:,200:]

state = data[:2,:].T
actual = np.zeros(state.shape)
actual[0,:] = state[0,:]
control = np.zeros((state.shape[0]-1,1))
ns2u = torch.load('invpen/invpen_nextstate2control.pth')
su2n = torch.load('invpen/invpen_statecontrol2next.pth')

# for ii in range(state.shape[0]-1):
#     ns = torch.Tensor(state[ii+1:ii+2,:])
#     a = torch.Tensor(actual[ii:ii+1,:])
#     netin = torch.cat((ns,a),dim=1)
#     control[ii,:] = ns2u(netin).detach().numpy().flatten()
#     # control[ii,0] = data[2,ii]
#     u = torch.Tensor(control[ii:ii+1,:])
#     netin = torch.cat((a,u),dim=1)
#     actual[ii+1,:] = su2n(netin).detach().numpy().flatten()

i = 0
k = 0.1

for ii in range(state.shape[0]-1):
    n = torch.Tensor(state[ii+1:ii+2,:])
    s = torch.Tensor(state[ii:ii+1,:])
    a = torch.Tensor(actual[ii:ii+1,:])
    netin = torch.cat((n,s),dim=1)
    control[ii,:] = ns2u(netin).detach().numpy().flatten() - ((a[0,i] - s[0,i]) * k).detach().numpy()
    u = torch.Tensor(control[ii:ii+1,:])
    # u[0,0] -= (a[0,1] - s[0,1]) / 0.1
    netin = torch.cat((a,u),dim=1)
    actual[ii+1,:] = su2n(netin).detach().numpy().flatten()


actual = actual.T
control = control.T

plt.figure()
plt.plot(data[0,:],label='th')
plt.plot(data[1,:],label='thd')
plt.plot(data[2,:-1],label='u')
plt.plot(actual[0,:],label='th net')
plt.plot(actual[1,:],label='thd net')
plt.plot(control[0,:],label='u net')
plt.legend()
plt.show()
