from rplidar import RPLidar
import hough_transform as hough
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


 # convert from Hough line to cartesian line (y = mx+b)
def hough_to_cartesian_line(rho, theta):
    if(theta == 0): # make infintessimally small so that 1/tan is always defined
        theta = 0.00000000001
    m = -math.cos(theta)/math.sin(theta)
    b = rho/math.sin(theta)
    return m, b

def linear_intersection(m1, b1, m2, b2):
    if (m1 == m2):
        m2 -= 0.00000000001
    x = (b2 - b1)/(m1 - m2)
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

def norm(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    n = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return n

def swap(A, B):
    temp = A
    A = B
    B = temp
    return A, B

# -- Label Data -- 
# Return: add a third column to 'arr' with index of each row
def label_data(arr):
    index_col = np.zeros([len(arr), 1])
    for i in range(len(arr)):
        index_col[i] = int(i)
    indexed_data = np.append(arr, index_col, 1)
    return indexed_data

# -- Identify Local Maxima --
# spacing: the number of datapoints to the left and right you look to determine if data[i] is a local maximum
def scan_local_maxima(data, spacing, frequency):
    range_maxima_lst = []
    for i in range(spacing, len(data)-spacing):
        max = True
        for j in range(1, spacing):
            if ((data[i][1] < data[i+j][1]) or 
                    (data[i][1] < data[i-j][1])):
                max = False
        if max == True:
            range_maxima_lst.append(data[i])      
    lidar_data_maxima = np.array(range_maxima_lst)
    return lidar_data_maxima


# -- Gets Data on Either Side of Maximum --
# gets lower and upper ranges while wrapping for values that cross 0deg
def upper_lower_Polar_Wrap(local_range, index, data):
    # wrap polar corrdinates
    if ((index - local_range) < 0): 
        overhang = -(index - local_range)
        wrap_ind = len(data) - overhang
        l_range = data[0:(index), :]
        l_range = np.concatenate((l_range, data[wrap_ind:(len(data)), :]))
        u_range = data[(index+1):(index+1 + local_range), :]
    elif((index + local_range) > len(data)-1):
        overhang = (index + local_range) - len(data)
        wrap_ind = overhang
        u_range = data[0:wrap_ind, :]
        u_range = np.concatenate((u_range, data[(index+1):(len(data)), :]))
        l_range = data[(index - local_range):(index), :]
    else:
        l_range = data[(index - local_range):(index), :]
        u_range = data[(index+1):(index+1 + local_range), :]
    return l_range, u_range

# -- Hough Transform Error Range --
# run hough transform on pts above and below the hough line to get a box representing error 
def hough_error(l_range, u_range, m_l, m_u, b_l, b_u):
    A_l_cart = [[0,0]]
    B_l_cart = [[0,0]]
    A_l_polar = [[0,0]]
    B_l_polar = [[0,0]]
    for k in range(len(l_range)):
        x_l = l_range[k][1]*math.cos(math.radians(l_range[k][0])) # x val = r*cos(theta)
        y_l = l_range[k][1]*math.sin(math.radians(l_range[k][0])) # y val = r*sin(theta)
        if((m_l*x_l + b_l) < y_l): # if for line x = x_l
            A_l_cart.append([x_l, y_l])
            A_l_polar.append([l_range[k][0], l_range[k][1]])
        else: 
            B_l_cart.append([x_l, y_l])
            B_l_polar.append([l_range[k][0], l_range[k][1]])
    A_u_cart = [[0,0]]
    B_u_cart = [[0,0]]
    A_u_polar = [[0,0]]
    B_u_polar = [[0,0]]
    for k in range(len(u_range)):
        x_u = u_range[k][1]*math.cos(math.radians(u_range[k][0])) # x val = r*cos(theta)
        y_u = u_range[k][1]*math.sin(math.radians(u_range[k][0])) # y val = r*sin(theta)
        if((m_u*x_u + b_u) < y_u): # if for line x = x_u
            A_u_cart.append([x_u, y_u])
            A_u_polar.append([u_range[k][0], u_range[k][1]])
        else: 
            B_u_cart.append([x_u, y_u])
            B_u_polar.append([u_range[k][0], u_range[k][1]])
    A_u_cart = np.array(A_u_cart)
    B_u_cart = np.array(B_u_cart)
    A_l_cart = np.array(A_l_cart)
    B_l_cart = np.array(B_l_cart)
    A_u_polar = np.array(A_u_polar)
    B_u_polar = np.array(B_u_polar)
    A_l_polar = np.array(A_l_polar)
    B_l_polar = np.array(B_l_polar)
    A_l_theta, A_l_rho, A_l_acc_max = hough.hough_line(A_l_polar, 2)
    B_l_theta, B_l_rho, B_l_acc_max = hough.hough_line(B_l_polar, 2)
    A_u_theta, A_u_rho, A_u_acc_max = hough.hough_line(A_u_polar, 2)
    B_u_theta, B_u_rho, B_u_acc_max = hough.hough_line(B_u_polar, 2)
    m_A_u, b_A_u = hough_to_cartesian_line(A_u_rho, A_u_theta)
    m_B_u, b_B_u = hough_to_cartesian_line(B_u_rho, B_u_theta)
    m_A_l, b_A_l = hough_to_cartesian_line(A_l_rho, A_l_theta)
    m_B_l, b_B_l = hough_to_cartesian_line(B_l_rho, B_l_theta)

    # find 4 intersection points to make grid (1:[Au, Al], 2:[Au, Bl], 3:[Bu, Al], 4:[Bu, Bl])
    int_x_AuAl, int_y_AuAl = linear_intersection(m_A_u, b_A_u, m_A_l, b_A_l)
    int_x_AuBl, int_y_AuBl = linear_intersection(m_A_u, b_A_u, m_B_l, b_B_l)
    int_x_BuAl, int_y_BuAl = linear_intersection(m_B_u, b_B_u, m_A_l, b_A_l)
    int_x_BuBl, int_y_BuBl = linear_intersection(m_B_u, b_B_u, m_B_l, b_B_l)

    error_range = np.array([[int_x_AuAl, int_y_AuAl], [int_x_AuBl, int_y_AuBl], [int_x_BuAl, int_y_BuAl], [int_x_BuBl, int_y_BuBl]])
    return error_range, A_l_cart, A_u_cart, B_l_cart, B_u_cart, m_A_u, b_A_u, m_B_u, b_B_u, m_A_l, b_A_l, m_B_l, b_B_l

def plot_deviation(A_u_cart, B_u_cart, A_l_cart, B_l_cart, m_A_u, b_A_u, m_B_u, b_B_u, m_A_l, b_A_l, m_B_l, b_B_l):
    plt.rcParams["figure.figsize"] = [7, 7]
    plt.rcParams["figure.autolayout"] = True

    # plt.scatter(A_u_cart[:, 0], A_u_cart[:, 1], c='red') 
    # plt.scatter(B_u_cart[:, 0], B_u_cart[:, 1], c='blue') 
    # plt.scatter(A_l_cart[:, 0], A_l_cart[:, 1], c='black') 
    # plt.scatter(B_l_cart[:, 0], B_l_cart[:, 1], c='green') 

    plt.scatter(A_u_cart[:, 0], A_u_cart[:, 1], c='red') 
    plt.plot([0, -b_A_u/m_A_u], [b_A_u, 0], color='red')
    
    plt.scatter(B_u_cart[:, 0], B_u_cart[:, 1], c='blue') 
    plt.plot([0, -b_B_u/m_B_u], [b_B_u, 0], color='blue')

    plt.scatter(A_l_cart[:, 0], A_l_cart[:, 1], c='black') 
    plt.plot([0, -b_A_l/m_A_l], [b_A_l, 0], color='black')

    plt.scatter(B_l_cart[:, 0], B_l_cart[:, 1], c='green') 
    plt.plot([0, -b_B_l/m_B_l], [b_B_l, 0], color='green')
    # plt.scatter(error_range[:, 0], error_range[:, 1], c='orange')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Deviation")
    # Display the plot
    plt.show() 

def plot_maxima_search_area(l_range, u_range):
     # -- Plot Maxima Search Area (lower and upper) -- 
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location("N")
    ax1.grid(True)
    for j in range(len(l_range)):
        ax1.plot(l_range[j][0]* -np.pi/180, l_range[j][1],'r+')
    for j in range(len(u_range)):
        ax1.plot(u_range[j][0]* -np.pi/180, u_range[j][1],'b+')    
    plt.show() 


def get_Lidar_scan():
    # sets the com port of RPLidar:
    lidar = RPLidar('COM4')    # '/dev/ttyS3' for WSL 
    scans = []
    num_datapoints = 0
        # need several scans (i.e. 10) for complete point cloud
    for i, scan in enumerate(lidar.iter_scans()):
            # print('%d: Got %d measures' % (i, len(scan)))
        scans = scans + [list(item) for item in scan] # convert tuple to list
        num_datapoints += len(scan)
        if i > 8:
            break
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
        # fill data array [theta(deg), r(mm)]
    scan_data = np.array(scans)[:,1:]
    return scan_data


''' 
Finds the translational and rotational matrices to transform points in A to points in B
Input:  
    two sets of corners, A and B, which share some points, altered by a transformation
Return: 
    dx - x-component of translation
    dy - y-component of translation
    dtheta - angle of rotation
'''
def match_transform(A, B):
    # make sure the cardinality of both sets A and B are the same
    while(len(A) != len(B)): # make sure A is longer than B
        min_lst = []
        if len(A) < len(B):
            A, B = swap(A, B)
        for i in range(len(A)):
            min_dist = 10000000
            min_index = 0    
            for j in range(len(B)):
                dist = norm(A[i], B[j])
                if dist < min_dist:
                    min_dist = dist
            min_lst.append(min_dist)
        max_min_index = min_lst.index(max(min_lst))
        # print(min_lst)
        del A[max_min_index] # delete the extraneous point
    
    T, R1, t1 = icp.best_fit_transform(np.array(A), np.array(B))
    d_x = t1[0]
    d_y = t1[1]
    d_theta = math.acos(R1[0][0])
    return d_x, d_y, d_theta


def main():    
    num_iter = 0
    corner_data_buffer = []
    state = [0, 0, 0] # state variables defined as x, y, theta

    # Scans every 5 seconds and measures the change in state
    while True:
        # -- Choose Data File, Sort, & Index --
        lidar_data = get_Lidar_scan()
        lidar_data = lidar_data[lidar_data[:, 0].argsort()]
        indexed_data = label_data(lidar_data)
        # print(lidar_data)
        lidar_data_maxima = scan_local_maxima(indexed_data, 30, 1)

        # -- Search Data Around Maxima -- 
        # inspect area left and right of the local maxima and pass to hough transform
        maxima_inspection_range = 30
        flip_angle = -360
        corners_list = []
        for i in range(len(lidar_data_maxima)):
            index = int(lidar_data_maxima[i][2])
            # print("\nInspecting possible corner at", -(lidar_data_maxima[i][0]+flip_angle), "degrees...")

            # get LOWER and UPPER ranges 
            l_range, u_range = upper_lower_Polar_Wrap(maxima_inspection_range, index, lidar_data)
            l_theta, l_rho, l_acc_max = hough.hough_line(l_range, 2)
            u_theta, u_rho, u_acc_max = hough.hough_line(u_range, 2)
            # print("Lower angle (red):", np.rad2deg(l_theta), l_rho)
            # print("Upper angle (blue):", np.rad2deg(u_theta), u_rho)
            
            m_u, b_u = hough_to_cartesian_line(u_rho, u_theta)
            m_l, b_l = hough_to_cartesian_line(l_rho, l_theta)
            x_intersection, y_intersection = linear_intersection(m_l, b_l, m_u, b_u)
            edge_angle = abs(np.rad2deg(l_theta) - np.rad2deg(u_theta))
            # print("lower line:", m_l, "x + ", b_l)
            # print("upper line:", m_u, "x + ", b_u)

            # error_range, A_l_cart, A_u_cart, B_l_cart, B_u_cart, m_A_u, b_A_u, m_B_u, b_B_u, m_A_l, b_A_l, m_B_l, b_B_l = hough_error(l_range, u_range, m_l, m_u, b_l, b_u)
            
            if(edge_angle < 100 and edge_angle > 80 and (u_acc_max + l_acc_max) > 13): # higher acc_max value means that the line has more clarity
                # print("***********  Corner spotted! At", -(lidar_data_maxima[i][0]+flip_angle), "degrees. **************")
                corners_list.append([y_intersection, x_intersection])
                # plot_deviation(A_u_cart, B_u_cart, A_l_cart, B_l_cart, m_A_u, b_A_u, m_B_u, b_B_u, m_A_l, b_A_l, m_B_l, b_B_l)
                # plot_maxima_search_area(l_range, u_range)
            # else: print("No corner at", -(lidar_data_maxima[i][0]+flip_angle), "degrees.")
        # print(corners_list)
        
        if corners_list: # if list is not empty 
            corner_data_buffer.append(corners_list)
            print("Number of Corners:", len(corners_list))
            if num_iter > 0:
                d_x, d_y, d_theta = match_transform(corner_data_buffer[-2], corner_data_buffer[-1])
                print("\ndx", d_x, "\ndy", d_y, "\ndTheta", d_theta)
                state[0] += d_x
                state[1] += d_y
                state[2] += d_theta
                print("state", state)
        else: print("not enough corners")
        num_iter += 1
                
        if keyboard.is_pressed("q"):
            # Key was pressed
            break
        
        print("Sleeping...zzz...")
        time.sleep(5)
    
if __name__ == "__main__":
    main()

