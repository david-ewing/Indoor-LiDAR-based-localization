from rplidar import RPLidar
import hough_transform as hough
import Lidar_SLAM

import sys

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
import csv  
import math

# -- Settings --
np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=sys.maxsize)

DMAX = 200
IMIN = 0
IMAX = 50

outlier_detection_tolerance = 20 # (mm)
outlier_detection_quality = 90 # (%)
num_lidar_scans = 8 # (scans)
local_maxima_scan_frequency = 1 # (datapoint/datapoint)
local_maxima_scan_range = 10 # (datapoints) - each local maxima must be the maximum of this number of datapoints on either side
hough_inspection_range = 30 # (datapoints) - this number of data points are sent to HT on either side of maxima



# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# ax.set_theta_zero_location("N")
# ax.grid(True)
# ax.set_ylim(1000, 3000)


 # convert from Hough line to cartesian line (y = mx+b)
def hough_to_cartesian_line(rho, theta):
    if(theta == 0): # make infintessimally small so that 1/tan is always defined
        theta = 0.00000000001
    m = -math.cos(theta)/math.sin(theta)
    b = rho/math.sin(theta)
    return m, b

def linear_intersection(m1, b1, m2, b2):
    if((m1 - m2) == 0):
        x = (b2 - b1)/0.000000001
    else: x = (b2 - b1)/(m1 - m2)
    y = m1*x + b1
    return x, y

def polar_to_cartesian_arr(arr):
    cart_arr = np.zeros([len(arr), 2]) # L by 2 np array
    for i in range(len(arr)):
        theta = arr[i][0]
        r = arr[i][1]
        cart_arr[i][0] = r*math.cos(math.radians(theta)) # x val
        cart_arr[i][1] = r*math.sin(math.radians(theta)) # y val
    return cart_arr



# -- Choose Data File & Sort--
data_name = input("Input data file name: ")
lidar_data = np.load('Data/' + data_name + '.npy')
lidar_data = lidar_data[lidar_data[:, 0].argsort()]
data_size = len(lidar_data)
print("HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
# -- Label Data -- 
index_col = np.zeros([data_size, 1])
for i in range(data_size):
    index_col[i] = int(i)
indexed_data = np.append(lidar_data, index_col, 1)

# -- Choose Data Subset --
min_angle = 0 # 22.2
max_angle = 360 # 45
lidar_data_sample_lst = []
lidar_data_sample_sz = 0
for i in range(data_size):
    if (indexed_data[i][0] < max_angle) and (indexed_data[i][0] > min_angle):
        lidar_data_sample_lst.append(indexed_data[i])
        lidar_data_sample_sz +=1
lidar_data_sample = np.array(lidar_data_sample_lst)


fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
ax1.set_theta_zero_location("N")
ax1.grid(True)
for j in range(len(lidar_data_sample)):
    ax1.plot(lidar_data_sample[j][0]* -np.pi/180, lidar_data_sample[j][1],'r+')  
plt.show() 

# -- Plot Data Subset -- 
# print('plotting')
# for i in range(lidar_data_sample_sz):
#    plt.plot(lidar_data_sample[i][0]* np.pi/180, lidar_data_sample[i][1], 'r+') 
# plt.show()  

# -- Identify Local Maxima --
range_maxima_lst = []
lidar_data_maxima_sz = 0
spacing = 10 # the number of datapoints to the left and right you look to determine if data[i] is a local maximum
for i in range(spacing, lidar_data_sample_sz-spacing):
    max = True
    for j in range(1, spacing):
        if ((lidar_data_sample[i][1] < lidar_data_sample[i+j][1]) or 
                (lidar_data_sample[i][1] < lidar_data_sample[i-j][1])):
            max = False
    if max == True:
        range_maxima_lst.append(lidar_data_sample[i])
        lidar_data_maxima_sz +=1        

lidar_data_maxima = np.array(range_maxima_lst)
# print(lidar_data_maxima)
# print(lidar_data_maxima_sz)

# -- Search Data Around Maxima -- 
# inspect area left and right of the local maxima and pass to hough transform
maxima_inspection_range = 40
flip_angle = -360
corners_lst = []
for i in range(lidar_data_maxima_sz):
    index = int(lidar_data_maxima[i][2])
    outlier = False
    # print("\nInspecting possible corner at", -(lidar_data_maxima[i][0]+flip_angle), "degrees...")

     # wrap polar corrdinates
    if ((index - maxima_inspection_range) < 0): 
        overhang = -(index - maxima_inspection_range)
        wrap_ind = data_size - overhang
        l_range = lidar_data[0:(index), :]
        l_range = np.concatenate((l_range, lidar_data[wrap_ind:(data_size), :]))
        u_range = lidar_data[(index+1):(index+1 + maxima_inspection_range), :]
    elif((index + maxima_inspection_range) > data_size-1):
        overhang = (index + maxima_inspection_range) - data_size
        wrap_ind = overhang
        u_range = lidar_data[0:wrap_ind, :]
        u_range = np.concatenate((u_range, lidar_data[(index+1):(data_size), :]))
        l_range = lidar_data[(index - maxima_inspection_range):(index), :]
    else:
        l_range = lidar_data[(index - maxima_inspection_range):(index), :]
        u_range = lidar_data[(index+1):(index+1 + maxima_inspection_range), :]
    
    l_range_cart = hough.polar_to_cartesian_arr(l_range[:, 0], l_range[:, 1])
    u_range_cart = hough.polar_to_cartesian_arr(u_range[:, 0], u_range[:, 1])

    # LOWER RANGE
    l_theta, l_rho, l_acc_max = hough.hough_line(l_range_cart, 2)          # only theta neccessary to show orthogonality
    # print("Lower angle (red):", np.rad2deg(l_theta), l_rho)

    # UPPER RANGE
    u_theta, u_rho, u_acc_max = hough.hough_line(u_range_cart, 2)
    # print("Upper angle (blue):", np.rad2deg(u_theta), u_rho)
    
    m_u, b_u = hough_to_cartesian_line(u_rho, u_theta)
    m_l, b_l = hough_to_cartesian_line(l_rho, l_theta)
    x_intersection, y_intersection = linear_intersection(m_l, b_l, m_u, b_u)

    # error_range, A_l_cart, A_u_cart, B_l_cart, B_u_cart, m_A_u, b_A_u, m_B_u, b_B_u, m_A_l, b_A_l, m_B_l, b_B_l = hough_error(l_range, u_range, m_l, m_u, b_l, b_u)
   
    edge_angle = abs(np.rad2deg(l_theta) - np.rad2deg(u_theta))

    if(Lidar_SLAM.outlier_detected(m_u, b_u, u_cartesian, outlier_detection_tolerance, outlier_detection_quality) 
            or Lidar_SLAM.outlier_detected(m_l, b_l, l_cartesian, outlier_detection_tolerance, outlier_detection_quality)):
        print("Outlier Rejected")
        outlier = True

    print(outlier)

    print(l_theta/math.pi*180, u_theta/math.pi*180)
    # print("   ", edge_angle, "degree edge")
    if(edge_angle < 100 and edge_angle > 80 and (u_acc_max + l_acc_max) > 13): # higher acc_max value means that the line has more clarity
        
        print("***********  Corner spotted! At", -(lidar_data_maxima[i][0]+flip_angle), "degrees. **************")
        
        # corners_lst.append([-1*lidar_data_maxima[i][1]*math.sin(-math.radians(lidar_data_maxima[i][0]+flip_angle)), lidar_data_maxima[i][1]*math.cos(-math.radians(lidar_data_maxima[i][0]+flip_angle))])
        corners_lst.append([y_intersection, x_intersection])


        # -- Plot Maxima Search Area (lower and upper) -- 
        fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
        ax1.set_theta_zero_location("N")
        ax1.grid(True)
        for j in range(len(l_range)):
            ax1.plot(l_range[j][0]* -np.pi/180, l_range[j][1],'r+')
        for j in range(len(u_range)):
            ax1.plot(u_range[j][0]* -np.pi/180, u_range[j][1],'b+')    
        plt.show() 

    # else: print("No corner at", -(lidar_data_maxima[i][0]+flip_angle), "degrees.")

    
    
        

    #  # -- Plot Maxima Search Area (lower and upper) -- 
    # fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax1.set_theta_zero_location("N")
    # ax1.grid(True)
    # for j in range(len(l_range)):
    #     ax1.plot(l_range[j][0]* -np.pi/180, l_range[j][1],'r+')
    # for j in range(len(u_range)):
    #     ax1.plot(u_range[j][0]* -np.pi/180, u_range[j][1],'b+')    
    # plt.show() 

print(corners_lst)

