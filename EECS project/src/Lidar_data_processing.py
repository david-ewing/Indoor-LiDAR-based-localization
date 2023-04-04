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

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# ax.set_theta_zero_location("N")
# ax.grid(True)
# ax.set_ylim(1000, 3000)


# -- Choose Data File & Sort--
data_name = input("Input data file name: ")
lidar_data = np.load('Data/' + data_name + '.npy')
lidar_data = lidar_data[lidar_data[:, 0].argsort()]
data_size = len(lidar_data)

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
maxima_inspection_range = 50
flip_angle = -360
corners_lst = []
for i in range(lidar_data_maxima_sz):
    index = int(lidar_data_maxima[i][2])
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
    
    l_theta, l_rho, l_acc_max = hough.hough_line(l_range, 2)          # only theta neccessary to show orthogonality
    # print("Lower angle (red):", np.rad2deg(l_theta), l_rho)

    # upper range
    u_theta, u_rho, u_acc_max = hough.hough_line(u_range, 2)
    # print("Upper angle (blue):", np.rad2deg(u_theta), u_rho)
    
    # np.savetxt("l_accumulator", l_accumulator)
    # np.savetxt("u_accumulator", u_accumulator)

    # convert from Hough line to cartesian line (y = mx+b)
    if(u_theta == 0): # make infintessimally small so that 1/tan is always defined
        u_theta = 0.00000000001
    m_u = -math.cos(u_theta)/math.sin(u_theta)
    b_u = u_rho/math.sin(u_theta)

    # m_l = -math.tan2(l_theta)/math.sin(l_theta)
    if(l_theta == 0): 
        l_theta = 0.00000000001
    m_l = -math.cos(l_theta)/math.sin(l_theta)
    b_l = l_rho/math.sin(l_theta)
    if (m_l == m_u):
        m_u -= 0.00000000001
    x_intersection = (b_u - b_l)/(m_l - m_u)
    y_intersection = m_l*x_intersection + b_l

    # print("lower line:", m_l, "x + ", b_l)
    # print("upper line:", m_u, "x + ", b_u)


    # run hough transform on pts above and below the hough line 
    A_l = [[0,0]]
    B_l = [[0,0]]
    for k in range(len(l_range)):
        x_l = l_range[k][1]*math.cos(math.radians(l_range[k][0])) # x val = r*cos(theta)
        y_l = l_range[k][1]*math.sin(math.radians(l_range[k][0])) # y val = r*sin(theta)
        if((m_l*x_l + b_l) < y_l): # if for line x = x_l
            A_l.append([x_l, y_l])
        else: B_l.append([x_l, y_l])
    
    A_u = [[0,0]]
    B_u = [[0,0]]
    for k in range(len(u_range)):
        x_u = u_range[k][1]*math.cos(math.radians(u_range[k][0])) # x val = r*cos(theta)
        y_u = u_range[k][1]*math.sin(math.radians(u_range[k][0])) # y val = r*sin(theta)
        if((m_u*x_u + b_u) < y_u): # if for line x = x_u
            A_u.append([x_u, y_u])
        else: B_u.append([x_u, y_u])

    A_u_data = np.array(A_u)
    B_u_data = np.array(B_u)
    A_l_data = np.array(A_l)
    B_l_data = np.array(B_l)

    edge_angle = abs(np.rad2deg(l_theta) - np.rad2deg(u_theta))
    # print("   ", edge_angle, "degree edge")
    if(edge_angle < 100 and edge_angle > 80 and (u_acc_max + l_acc_max) > 13): # acc_max value describes the certainty that the hough transfor is accurate
        print("***********  Corner spotted! At", -(lidar_data_maxima[i][0]+flip_angle), "degrees. **************")
        corners_lst.append([-1*lidar_data_maxima[i][1]*math.sin(-math.radians(lidar_data_maxima[i][0]+flip_angle)), lidar_data_maxima[i][1]*math.cos(-math.radians(lidar_data_maxima[i][0]+flip_angle))])
        corners_lst.append([y_intersection, x_intersection])

        plt.rcParams["figure.figsize"] = [7, 7]
        plt.rcParams["figure.autolayout"] = True
        plt.scatter(A_u_data[:, 0], A_u_data[:, 1], c='red') 
        plt.scatter(B_u_data[:, 0], B_u_data[:, 1], c='blue') 
        plt.scatter(A_l_data[:, 0], A_l_data[:, 1], c='black') 
        plt.scatter(B_l_data[:, 0], B_l_data[:, 1], c='green') 
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Deviation")
        # Display the plot
        plt.show() 

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

# print(corners_lst[0], corners_lst[2])
# print(corners_lst[1], corners_lst[3])
print(corners_lst)