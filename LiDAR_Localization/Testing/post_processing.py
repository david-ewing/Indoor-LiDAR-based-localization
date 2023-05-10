from rplidar import RPLidar
import hough_transform as hough
import Lidar_Lib as Lidary

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


# -- Tuning Variables --
outlier_detection_tolerance = 30 # (mm)
outlier_detection_quality = 75 # (%)
num_lidar_scans = 8 # (scans)
local_maxima_scan_frequency = 1 # (datapoint/datapoint)
local_maxima_scan_range = 10 # (datapoints) - each local maxima must be the maximum of this number of datapoints on either side
hough_inspection_range = 20 # (datapoints) - this number of data points are sent to HT on either side of maxima

def main():
    # -- Choose Data File & Sort --
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

    # -- Choose Data Subset (defaults to entire dataset) --
    min_angle = 0 # 22.2
    max_angle = 360 # 45
    lidar_data_sample_lst = []
    lidar_data_sample_sz = 0
    for i in range(data_size):
        if (indexed_data[i][0] < max_angle) and (indexed_data[i][0] > min_angle):
            lidar_data_sample_lst.append(indexed_data[i])
            lidar_data_sample_sz +=1
    lidar_data_sample = np.array(lidar_data_sample_lst)

    # -- Plot Data --
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location("N")
    ax1.grid(True)
    for j in range(len(lidar_data_sample)):
        ax1.plot(lidar_data_sample[j][0]* -np.pi/180, lidar_data_sample[j][1],'r+')  
    plt.show()  

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
    maxima_inspection_range = 30
    flip_angle = -360
    corners_lst = []
    for i in range(lidar_data_maxima_sz):
        index = int(lidar_data_maxima[i][2])
        outlier = False
        # print("\nInspecting possible corner at", -(lidar_data_maxima[i][0]+flip_angle), "degrees...")

        l_range, u_range = Lidary.upper_lower_Polar_Wrap(maxima_inspection_range, index, lidar_data)
        
        
        l_range_cart = Lidary.polar_to_cartesian_arr(l_range)
        u_range_cart = Lidary.polar_to_cartesian_arr(u_range)

        # LOWER RANGE
        l_theta, l_rho, l_acc_max = hough.hough_line(l_range_cart, 2)          # only theta neccessary to show orthogonality
        # print("Lower angle (red):", np.rad2deg(l_theta), l_rho)

        # UPPER RANGE
        u_theta, u_rho, u_acc_max = hough.hough_line(u_range_cart, 2)
        # print("Upper angle (blue):", np.rad2deg(u_theta), u_rho)
        
        m_u, b_u = Lidary.hough_to_cartesian_line(u_rho, u_theta)
        m_l, b_l = Lidary.hough_to_cartesian_line(l_rho, l_theta)
        x_intersection, y_intersection = Lidary.linear_intersection(m_l, b_l, m_u, b_u)

        # error_range, A_l_cart, A_u_cart, B_l_cart, B_u_cart, m_A_u, b_A_u, m_B_u, b_B_u, m_A_l, b_A_l, m_B_l, b_B_l = hough_error(l_range, u_range, m_l, m_u, b_l, b_u)
    
        edge_angle = abs(np.rad2deg(l_theta) - np.rad2deg(u_theta))

        # print(l_theta/math.pi*180, u_theta/math.pi*180)
        # print("   ", edge_angle, "degree edge")
        if(edge_angle < 100 and edge_angle > 80 and (u_acc_max + l_acc_max) > 13): # higher acc_max value means that the line has more clarity
            
            if(Lidary.outlier_detected(m_u, b_u, u_range_cart, outlier_detection_tolerance, outlier_detection_quality) 
                or Lidary.outlier_detected(m_l, b_l, l_range_cart, outlier_detection_tolerance, outlier_detection_quality)):
                print("Outlier Rejected")
                outlier = True

            print("***********  Corner spotted! At", -(lidar_data_maxima[i][0]+flip_angle), "degrees. **************")
            # print("outlier:", outlier)
            # corners_lst.append([-1*lidar_data_maxima[i][1]*math.sin(-math.radians(lidar_data_maxima[i][0]+flip_angle)), lidar_data_maxima[i][1]*math.cos(-math.radians(lidar_data_maxima[i][0]+flip_angle))])
            corners_lst.append([y_intersection, x_intersection])
            
            # -- Plot Maxima Search Area (lower and upper) & Hough Lines -- 
            fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
            ax1.set_theta_zero_location("N")
            ax1.grid(True)
            
            min_x_l = np.min(l_range_cart[:, 0])
            max_x_l = np.min(l_range_cart[:, 0])
            x_l = np.linspace(min_x_l-200, max_x_l+200, 100)
            y_l = m_l*x_l+b_l
            
            ax1.plot(-np.arctan2(y_l,x_l), np.sqrt(x_l**2+y_l**2), '-g', label='Lower Hough Line')

            min_x_u = np.min(u_range_cart[:, 0])
            max_x_u = np.min(u_range_cart[:, 0])
            x_u = np.linspace(min_x_u-200, max_x_u+200, 100)
            y_u = m_u*x_u+b_u
            ax1.plot(-np.arctan2(y_u,x_u), np.sqrt(x_u**2+y_u**2), '-y', label='Upper Hough Line')

            # l_range_cart2 = pol2cart(l_range[:, 1], phi)

            ax1.plot(-l_range[:, 0]* np.pi/180, l_range[:, 1],'r+')
            ax1.plot(-u_range[:, 0]* np.pi/180, u_range[:, 1],'b+')
            
            # ax1.plot(-np.arctan2(l_range_cart[:, 1], l_range_cart[:, 0]), np.sqrt(l_range_cart[:, 0]**2+l_range_cart[:, 1]**2),'g+')   
            # ax1.plot(-np.arctan2(u_range_cart[:, 1], u_range_cart[:, 0]), np.sqrt(u_range_cart[:, 0]**2+u_range_cart[:, 1]**2),'y+')   

            plt.show() 

        # else: print("No corner at", -(lidar_data_maxima[i][0]+flip_angle), "degrees.")

    print(corners_lst)



if __name__ == "__main__":
    main()
