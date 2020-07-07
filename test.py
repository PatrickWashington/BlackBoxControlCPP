import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# print('hello from test.py')

cmd = './run invpen net invpen/invpen_network.csv test invpen/invpen_ilqr_result.csv'
os.system(cmd)

# print('I am here now')


filename = 'invpen/net_invpen_test_result.csv'
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

filename = 'invpen/invpen_ilqr_result.csv'
with open(filename) as file:
    filereader = csv.reader(file,delimiter=',')
    ii = 0
    for row in filereader:
        if ii == 0:
            numrow = int(row[0])
            numcol = int(row[1])
            data2 = np.zeros((numrow,numcol))
        else:
            for jj in range(numcol):
                data2[ii-1,jj] = float(row[jj])
        ii += 1

plt.figure()
plt.plot(data[0,:],label='th1')
plt.plot(data[1,:],label='thd1')
plt.plot(data[2,:],label='u')
plt.plot(data2[0,:],label='th1')
plt.plot(data2[1,:],label='thd1')
plt.legend()
plt.show()
