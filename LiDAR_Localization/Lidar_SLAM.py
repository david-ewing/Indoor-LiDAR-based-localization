from rplidar import RPLidar
import hough_transform as hough
import Lidar_Lib as Lidary
import icp
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
import csv  
import math
import time
import keyboard
from tabulate import tabulate

# -- Settings --
np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=sys.maxsize)

DMAX = 200
IMIN = 0
IMAX = 50


def main():    
    # Tuning Parameters:
    outlier_detection_tolerance = 20 # (mm)
    outlier_detection_quality = 90 # (%)
    num_lidar_scans = 8 # (scans)
    local_maxima_scan_frequency = 1 # (datapoint/datapoint)
    local_maxima_scan_range = 10 # (datapoints) - each local maxima must be the maximum of this number of datapoints on either side
    hough_inspection_range = 30 # (datapoints) - this number of data points are sent to HT on either side of maxima
    duplicate_corner_range = 10 # (mm) - remove two corners if they are within this distance from each other
    coner_angle_tolerance = 10 # (degrees) 

    # Initialize Variables
    num_iter = 0
    corner_data_buffer = []
    state = [0, 0, 0] # state variables defined as x, y, theta
    state_list = [[0,0,0]]
    st_ind = 0
    lidar, lidar_iterator = Lidary.init_Lidar_scan() # initialize continuous scan

    table = [["Index", "Time", "state", "dX", "dY", "dTheta", "Num Corners"]]
    time_offset = time.time()

    # Scans every continuously and measures the change in state
    while True:
        # -- Choose Data File, Sort, Index, Get Local Maxima --
        scan_time = time.time() - time_offset
        lidar_data = Lidary.get_Lidar_scan(lidar_iterator, num_lidar_scans)
        lidar_data = lidar_data[lidar_data[:, 0].argsort()]
        indexed_data = Lidary.label_data(lidar_data)
        lidar_data_maxima = Lidary.scan_local_maxima(indexed_data, local_maxima_scan_range, local_maxima_scan_frequency)
        # Lidary.plot_data(lidar_data)
    
        # -- Search Data Around Maxima and Verify Corners -- 
        # inspect area left and right of the local maxima and pass to hough transform
        flip_angle = -360
        corners_list = []
        for i in range(len(lidar_data_maxima)):
            outlier = False
            index = int(lidar_data_maxima[i][2])
            # print("\nInspecting possible corner at", -(lidar_data_maxima[i][0]+flip_angle), "degrees...")

            # Get left, l, and right, r, ranges 
            l_range, r_range = Lidary.upper_lower_Polar_Wrap(hough_inspection_range, index, lidar_data)
            
            # Convert to cartesian coords
            l_cartesian = Lidary.polar_to_cartesian_arr(l_range)
            r_cartesian = Lidary.polar_to_cartesian_arr(r_range)
            l_theta, l_rho, l_acc_max = hough.hough_line(l_cartesian, 2)
            r_theta, r_rho, r_acc_max = hough.hough_line(r_cartesian, 2)
            m_r, b_r = Lidary.hough_to_cartesian_line(r_rho, r_theta)
            m_l, b_l = Lidary.hough_to_cartesian_line(l_rho, l_theta)

            # Calculate intersection point of l and r Hough lines
            x_intersection, y_intersection = Lidary.linear_intersection(m_l, b_l, m_r, b_r)
            edge_angle = abs(np.rad2deg(l_theta) - np.rad2deg(r_theta))
            
            # Scan for outliers post-Hough transform
            if(Lidary.outlier_detected(m_r, b_r, r_cartesian, outlier_detection_tolerance, outlier_detection_quality) 
             or Lidary.outlier_detected(m_l, b_l, l_cartesian, outlier_detection_tolerance, outlier_detection_quality)):
                # print("Outlier Rejected")
                outlier = True 
            
            # Calculation of Hough deviation
            # error_range, A_l_cart, A_r_cart, B_l_cart, B_r_cart, m_A_r, b_A_r, m_B_r, b_B_r, m_A_l, b_A_l, m_B_l, b_B_l = Lidary.hough_error(l_range, r_range, m_l, m_r, b_l, b_r)
            
            if((edge_angle < 100) and (edge_angle > 80) and ((r_acc_max + l_acc_max) > 10) and not(outlier)): # higher acc_max value means that the line has more clarity
                corners_list.append([y_intersection, x_intersection, m_l, m_r])
                
                # Various plots for debugging:
                # print("***********  Corner spotted! At", -(lidar_data_maxima[i][0]+flip_angle), "degrees. **************")
                # Lidary.plot_deviation(A_r_cart, B_r_cart, A_l_cart, B_l_cart, m_A_r, b_A_r, m_B_r, b_B_r, m_A_l, b_A_l, m_B_l, b_B_l)
                # Lidary.plot_maxima_search_area(l_range, r_range)        
                # Lidary.plot_search_and_hough(l_range, r_range, l_cartesian, r_cartesian, m_l, b_l, m_r, b_r)

        # Scan for and remove duplicates (within duplicate_corner_range - default 10mm):
        del_list = []
        for j in range(len(corners_list)):
            for k in range(len(corners_list)):
                if not(j == k) and Lidary.norm([corners_list[j][0], corners_list[j][0]], [corners_list[k][0], corners_list[k][0]]) < duplicate_corner_range:
                    del_list.append(k) 
        for j in range(len(del_list), 0, -1):
            del corners_list[del_list[j]]
        
        # --- Add corners to Sequential Data Buffer and Find Transform ---
        if corners_list: # if list is not empty 
            corner_data_buffer.append(corners_list)
            print("Number of Corners:", len(corners_list))
            if num_iter > 1: # and (len(corner_data_buffer[-1]) - len(corner_data_buffer[-2]))/2 < len(corner_data_buffer[-2]): # check that the additional number of corners is not too drastic between datasets
                
                # Run ICP algorithm on the two most recent timesteps
                d_x, d_y, d_theta = Lidary.match_transform(corner_data_buffer[-2], corner_data_buffer[-1]) 

                # Update state variables (account for noise in stationary position)
                if(abs(d_x) < 5): d_x = 0
                if(abs(d_y) < 5): d_y = 0
                
                # Initialize Rotation variable to be triggered if the transformation appears to be only a rotation
                Rotation = False

                # Handling for the case of only one corner in the dataset
                #  - If 1 corner, find the change in orientation of that corner 
                #  - assumes there can only ever be just translation or just rotaiton
                if(len(corner_data_buffer[-2]) == 1 or len(corner_data_buffer[-1]) == 1):
                    if(len(corner_data_buffer[-1]) == 1):
                        singular_corner_data = corner_data_buffer[-1]
                        secondary_corner_data = corner_data_buffer[-2]
                    else:
                        singular_corner_data = corner_data_buffer[-2]
                        secondary_corner_data = corner_data_buffer[-1]

                    for c in range(len(secondary_corner_data)):  # checks to see if the distance to the corner remains constant between timesteps 
                        if(math.isclose(Lidary.norm([singular_corner_data[0][0], singular_corner_data[0][1]], [0,0]),
                            Lidary.norm([secondary_corner_data[c][0], secondary_corner_data[c][1]], [0,0]), rel_tol=0.05) and not math.isclose(singular_corner_data[0][0], secondary_corner_data[c][0], rel_tol=0.05)):
                            Rotation = True                 
                if Rotation:
                    d_theta = math.atan2(d_y, d_x)
                    d_x = 0
                    d_y = 0

                # Update state variables
                state[0] += d_x
                state[1] += d_y
                state[2] += d_theta

                # Update table
                state_list.append(state.copy())
                st_ind += 1
                table.append([num_iter, scan_time, state_list[st_ind], d_x, d_y, d_theta, len(corners_list)])
                print(tabulate(table, headers='firstrow'))

            num_iter += 1
            
        else: print("not enough corners")
        
        # Hold 'q' to exit program and disconnect rplidar 
        if keyboard.is_pressed("q"):
            break

    print(tabulate(table, headers='firstrow'))
    Lidary.disconnect_Lidar(lidar)
    
if __name__ == "__main__":
    main()

