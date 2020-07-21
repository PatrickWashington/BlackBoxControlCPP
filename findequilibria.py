import numpy as np
import matplotlib.pyplot as plt
import csv
import os

cmd = './run invpen net invpen/invpen_network.csv eq'
# cmd = './run invpen eq'
os.system(cmd)

# filename = 'invpen/invpen_eq.csv'
filename = 'invpen/invpen_net_eq.csv'
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
plt.plot((data[0,:]%(2*np.pi))*180/np.pi,data[1,:]*180/np.pi,'.')
plt.xlabel('theta')
plt.ylabel('thetadot')
plt.title('Equilibrium Points')
plt.show()