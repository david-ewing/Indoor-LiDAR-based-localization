from rplidar import RPLidar

import matplotlib.pyplot as plt
import numpy as np
import csv 

np.set_printoptions(suppress=True)

DMAX = 200
IMIN = 0
IMAX = 50

fig = plt.figure()
ax = plt.subplot(111, projection='polar')

ax.set_rmax(DMAX)
ax.grid(True)
ax.set_theta_zero_location("N")


# sets the com port of RPLidar:
lidar = RPLidar('COM3')    # '/dev/ttyS3' for WSL 

scans = []
num_datapoints = 0
# need several scans (i.e. 10) for complete point cloud
for i, scan in enumerate(lidar.iter_scans()):
   # print('%d: Got %d measures' % (i, len(scan)))
   num_datapoints += len(scan)
   scans.append(scan)
   if i > 8:
      break

lidar.stop()
lidar.stop_motor()
lidar.disconnect()

# complete data array [theta(deg), r(mm)]
scan_data = np.empty([num_datapoints, 2])

# fill data array
data_index = 0
for i in range(len(scans)):
   for j in range(len(scans[i])):
      scan_data[data_index][0] = scans[i][j][1]
      scan_data[data_index][1] = scans[i][j][2]
      data_index += 1

# save data to text file
data_name = input("Input test name: ")
np.save('Data/' + data_name, scan_data)
scan_data1 = np.load('Data/' + data_name + '.npy')

# plotting data
print('plotting')
for i in range(num_datapoints):
   plt.plot(scan_data1[i][0]* -np.pi/180, scan_data1[i][1], 'r+') 
plt.show()  

# save plot
fig.savefig('Data/' + data_name + '.png')

