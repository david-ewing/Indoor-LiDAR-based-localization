from rplidar import RPLidar
import hough_transform as hough

import sys

import matplotlib.pyplot as plt
import numpy as np
import csv  
import math

# -- Settings --
np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=sys.maxsize)

DMAX = 200
IMIN = 0
IMAX = 50

# -- Create and Plot Data Subset --
box_data = np.array([[-10, 10, 1],  [-9, 10, 1],    [-8, 10, 1],    [-7, 10, 1],    [-6, 10, 1],
                    [-5, 10,1],     [-4, 10, 1],    [-3, 10, 1],    [-2, 10, 1],    [-1, 10, 1],
                    [9, 10, 1],     [8, 10, 1],     [7, 10, 1],     [6, 10, 1],     [5, 10, 1],
                    [4, 10, 1],     [3, 10, 1],     [2, 10, 1],     [1, 10, 1],     [0, 10, 1], #top
                    [10, 10, 2],    [10, 9, 2],     [10, 8, 2],     [10, 7, 2],     [10, 6, 2],
                    [10, 5, 2],     [10, 4, 2],     [10, 3, 2],     [10, 2, 2],     [10, 1, 2],
                    [10, 0, 2],     [10, -10, 2],   [10, -9, 2],    [10, -8, 2],    [10, -7, 2],
                    [10, -6, 2],    [10, -5, 2],    [10, -4, 2],    [10, -3, 2],    [10, -2, 2],[10, -1, 2], #right
                    [-10, -10, 3],  [-9, -10, 3],   [-8, -10, 3],   [-7, -10, 3],   [-6, -10, 3],
                    [-5, -10, 3],   [-4, -10, 3],   [-3, -10, 3],   [-2, -10, 3],   [-1, -10, 3],
                    [9, -10, 3],    [8, -10, 3],    [7, -10, 3],    [6, -10, 3],    [5, -10, 3],
                    [4, -10, 3],    [3, -10, 3],    [2, -10, 3],    [1, -10, 3],    [0, -10, 3], #bottom
                    [-10, 9, 4],    [-10, 8, 4],    [-10, 7, 4],    [-10, 6, 4],    [-10, 5, 4],
                    [-10, 4, 4],    [-10, 3, 4],    [-10, 2, 4],    [-10, 1, 4],    [-10, 0, 4],
                    [-10, -9, 4],   [-10, -8, 4],   [-10, -7, 4],   [-10, -6, 4],   [-10, -5, 4],
                    [-10, -4, 4],   [-10, -3, 4],   [-10, -2, 4],   [-10, -1, 4]]) #left  

# box_data = np.array([[-10, 10],[-9, 10],[-8, 10],[-7, 10],[-6, 10],[-5, 10],[-4, 10],[-3, 10],[-2, 10],[-1, 10],[9, 10],[8, 10],[7, 10],[6, 10],[5, 10],[4, 10],[3, 10],[2, 10],[1, 10],[0, 10], #top
#                     [10, 10],[10, 9], [10, 8],[10, 7],[10, 6],[10, 5],[10, 4],[10, 3],[10, 2],[10, 1],[10, 0],[10, -10], [10, -9], [10, -8],[10, -7],[10, -6],[10, -5],[10, -4],[10, -3],[10, -2],[10, -1], #right
#                     [-10, -10],[-9, -10],[-8, -10],[-7, -10],[-6, -10],[-5, -10],[-4, -10],[-3, -10],[-2, -10],[-1, -10],[9, -10],[8, -10],[7, -10],[6, -10],[5, -10],[4, -10],[3, -10],[2, -10],[1, -10],[0, -10], #bottom
#                     [-10, 9],[-10, 8],[-10, 7],[-10, 6],[-10, 5],[-10, 4],[-10, 3],[-10, 2],[-10, 1],[-10, 0],[-10, -9], [-10, -8],[-10, -7],[-10, -6],[-10, -5],[-10, -4],[-10, -3],[-10, -2],[-10, -1]]) #left  

# plt.rcParams["figure.figsize"] = [5, 5]
# plt.rcParams["figure.autolayout"] = True
# plt.scatter(box_data[:, 0], box_data[:, 1], c=box_data[:, 2]) 
# plt.show() 

box_data_polar = np.zeros([len(box_data), 3]) # L by 2 np array
for i in range(len(box_data)):
    x_ = box_data[i][0]
    y_ = box_data[i][1]
    r = np.sqrt(x_**2 + y_**2)
    theta = (90-(np.arctan2(y_, x_)*180/math.pi))
    if(theta < 0):
        theta += 360
    box_data_polar[i][0] = theta
    box_data_polar[i][1] = r
    box_data_polar[i][2] = box_data[i][2]

# print(box_data_polar)
fig = plt.figure()
ax_b = fig.add_subplot(projection='polar')
ax_b.set_theta_zero_location("N")
ax_b.grid(True)
c = ax_b.scatter(box_data_polar[:, 0]*-math.pi/180, box_data_polar[:, 1], c=box_data_polar[:, 2])
plt.show()

# -- Sort Polar Data -- 
box_data_polar = box_data_polar[box_data_polar[:, 0].argsort()]
data_size = len(box_data_polar)
# print(box_data_polar)

index_col = np.zeros([data_size, 1])
for i in range(data_size):
    index_col[i] = int(i)
indexed_box_data = np.append(box_data_polar, index_col, 1)

# -- Identify Local Maxima --
range_maxima_lst = []
test_data_maxima_sz = 0
spacing = 5 # the number of datapoints to the left and right you look to determine if data[i] is a local maximum
for i in range(spacing, data_size-spacing):
    max = True
    for j in range(1, spacing):
        if ((indexed_box_data[i][1] < indexed_box_data[i+j][1]) or 
                (indexed_box_data[i][1] < indexed_box_data[i-j][1])):
            max = False
    if max == True:
        range_maxima_lst.append(indexed_box_data[i])
        test_data_maxima_sz +=1

test_data_maxima = np.array(range_maxima_lst)

# -- Search Data Around Maxima -- 
# inspect area left and right of the local maxima and pass to hough transform
maxima_inspection_range = 20
for i in range(test_data_maxima_sz):
    index = int(test_data_maxima[i][3])
    print("\nInspecting possible corner at", test_data_maxima[i][0], "degrees...")

    # wrap polar corrdinates
    if ((index - maxima_inspection_range) < 0): 
        overhang = -(index - maxima_inspection_range)
        wrap_ind = data_size - overhang
        l_range = box_data_polar[0:(index), :]
        l_range = np.concatenate((l_range, box_data_polar[wrap_ind:(data_size), :]))
        u_range = box_data_polar[(index+1):(index+1 + maxima_inspection_range), :]
    elif((index + maxima_inspection_range) > data_size-1):
        overhang = (index + maxima_inspection_range) - data_size
        wrap_ind = overhang
        u_range = box_data_polar[0:wrap_ind, :]
        u_range = np.concatenate((u_range, box_data_polar[(index+1):(data_size), :]))
        l_range = box_data_polar[(index - maxima_inspection_range):(index), :]
    else:
        l_range = box_data_polar[(index - maxima_inspection_range):(index), :]
        u_range = box_data_polar[(index+1):(index+1 + maxima_inspection_range), :]
    
    l_range_cart = hough.polar_to_cartesian_arr(l_range[:, 0], l_range[:, 1])
    u_range_cart = hough.polar_to_cartesian_arr(u_range[:, 0], u_range[:, 1])

    l_theta, l_rho, l_max_acc = hough.hough_line(l_range_cart, 2)          # only theta neccessary to show orthogonality
    # l_theta, l_rho, l_max_acc = hough.hough_line(l_range, 2)          # only theta neccessary to show orthogonality
    
    print("Lower angle (red):", np.rad2deg(l_theta))

    # upper range
    u_theta, u_rho, u_max_acc = hough.hough_line(u_range_cart, 2)
    # u_theta, u_rho, u_max_acc = hough.hough_line(u_range, 2)

    print("Upper angle (blue):", np.rad2deg(u_theta))
    
    # -- Plot Maxima Search Area (lower and upper) -- 
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location("N")
    ax1.grid(True)
    # ax1.set_ylim(1500, 3000)
    for j in range(len(l_range)):
        ax1.plot(l_range[j][0]* np.pi/180, l_range[j][1],'r+')
    for j in range(len(u_range)):
        ax1.plot(u_range[j][0]* np.pi/180, u_range[j][1],'b+')
    plt.show() 


    edge_angle = abs(np.rad2deg(l_theta) - np.rad2deg(u_theta))
    print("   ", edge_angle, "degree edge")
    if(edge_angle < 100 and edge_angle > 80):
        print("Corner spotted! At", test_data_maxima[i][0], "degrees.")
    else: print("No corner at", test_data_maxima[i][0], "degrees.")