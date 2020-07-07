import numpy as np
import os
import csv
import matplotlib.pyplot as plt

sysname = 'invpen net'
netfile = 'invpen/invpen_network.csv'
task = 'ilqr'
cmd = (' ').join(['./run',sysname,netfile,task])
os.system(cmd)


filename = 'invpen/net_invpen_ilqr_result.csv'
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

plt.figure()
plt.plot(data[0,:],label='th1')
plt.plot(data[1,:],label='thd1')
plt.plot(data[2,:],label='u')
plt.legend()
plt.show()