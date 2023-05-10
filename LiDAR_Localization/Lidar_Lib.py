import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
import csv  
import math
import icp

from rplidar import RPLidar
import hough_transform as hough


# COM port template:
# - Windows: 'COM#'
# - Linux: '/dev/ttyS#'
COM_PORT = 'COM4' 


def init_Lidar_scan():
    ''' Begin serial communication with RPlidar
    Output: lidar (object), lidar_iterator (list of lidar data) 
    '''
    # sets the com port of RPLidar:
    lidar = RPLidar(COM_PORT)    
    lidar_iterator = lidar.iter_scans()
    return lidar, lidar_iterator


def disconnect_Lidar(lidar):
    ''' Disconnect from RPLidar 
    '''
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()


def get_Lidar_scan(iterator, num_iter_scan):
    ''' Get some number of scans from RPLidar
    Input
    - iterator: used to iterate scans
    - num_iter_scans: number of scans desired for a single timestep
    Output: 
    - scan_data: all data from the iterated scans compiled in an numpy array
    '''
    scans = []
    num_datapoints = 0
    for i in range(num_iter_scan): # may need several scans (i.e. 10) for complete, dense point cloud
        scan = next(iterator)
        scans = scans + [list(item) for item in scan] # convert tuple to list
        num_datapoints += len(scan)
        if i > num_iter_scan:
            break
        # fill data array [theta(deg), r(mm)]
    scan_data = np.array(scans)[:,1:]
    return scan_data


def norm(pt1, pt2):
    ''' Find the distance between two points
    Input
     - pt1: (x1, y1)
     - pt2: (x2, y2)
    Output:
     - d: Euclidean distance between pt1 and pt2
    '''
    x1, y1 = pt1
    x2, y2 = pt2
    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return d

def swap(A, B):
    ''' Swap the values of two variables
    '''
    temp = A
    A = B
    B = temp
    return A, B


def hough_to_cartesian_line(rho, theta):
    ''' Convert from Hough line to cartesian line (y = mx+b)
    Input
     - rho: rho value of polar line equation
     - theta: theta value of polar line equation
    Output:
     - m: slope of cartesian line
     - b: y-intercept of cartesian line
    '''
    if(theta == 0): # make infintessimally small so that 1/tan is always defined
        theta = 0.00000000001
    m = -math.cos(theta)/math.sin(theta)
    b = rho/math.sin(theta)
    return m, b

def linear_intersection(m1, b1, m2, b2):
    ''' Find the point at which two lines intersect
    Input
     - m1, b1: equation of line 1
     - m2, b2: equation of line 2
    Output:
     - x, y: point of intersection
    '''
    if((m1 - m2) == 0):
        x = (b2 - b1)/0.000000001
    else: x = (b2 - b1)/(m1 - m2)
    y = m1*x + b1
    return x, y

def polar_to_cartesian_arr(arr):
    ''' Convert points from polar to cartesian form
    Input
     - arr: np array of n polar points (theta, rho)
    Output:
     - cart_arr: np array of n cartesian points (x, y)
    '''
    cart_arr = np.zeros([len(arr), 2]) # L by 2 np array
    for i in range(len(arr)):
        theta = math.radians(arr[i][0])
        r = arr[i][1]
        cart_arr[i][0] = r*math.cos(theta) # x val
        cart_arr[i][1] = r*math.sin(theta) # y val
    return cart_arr


def label_data(arr):
    ''' Label np array 
    Input
     - arr: np array
    Output:
     - indexed_data:  arr with a third column containing the index of each row
    '''
    index_col = np.zeros([len(arr), 1])
    for i in range(len(arr)):
        index_col[i] = int(i)
    indexed_data = np.append(arr, index_col, 1)
    return indexed_data


def scan_local_maxima(data, spacing, frequency):
    ''' Identify local maxima
    Input
     - data: np array of polar coordinates (theta, rho)
     - spacing: the number of datapoints to the left and right you look to determine if data[i] is a local maximum
     - frequency: the frequency of datapoints checked for maxima
    Output:
     - lidar_data_maxima: np array of all local maxima points
    '''
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


def outlier_detected(m, b, data, tolerance, quality):
    ''' Rejects outliers 
    Input:
     - m1, m2, b1, b2: line equations
     - data1, data2: data sets
     - tolerance: the acceptable error distance for each point to line
     - quality: the acceptable percentace of non-outlier points per set
    Output:
     - outlier: boolean var, indicates if input is really a corner or not
    '''
    outlier = False
    outlier_count = 0
    for i in range(len(data)):
        m_p = (-1/m) # find perpendicular line that intersects /w data pt
        b_p = data[i][1] - m_p*data[i][0]
        x_i, y_i = linear_intersection(m, b, m_p, b_p)
        dist = norm([x_i, y_i], [data[i][0], data[i][1]])
        if dist > tolerance:
            outlier_count += 1
    if (len(data) - outlier_count)/len(data)*100 < quality:
        outlier =True
    # print(outlier_count)
    return outlier


def upper_lower_Polar_Wrap(local_range, index, data):
    ''' gets left and right ranges surrounding a local maximum while wrapping for values that cross 0deg
    Input:
     - local_range: the number of points in range of the local max
     - index: index of local max
     - data: all scan data
    Output:
     - l_range: data left of max  
     - r_range: data right of max
    '''
    # wrap polar corrdinates
    if ((index - local_range) < 0): 
        overhang = -(index - local_range)
        wrap_ind = len(data) - overhang
        l_range = data[0:(index), :]
        l_range = np.concatenate((l_range, data[wrap_ind:(len(data)), :]))
        r_range = data[(index+1):(index+1 + local_range), :]
    elif((index + local_range) > len(data)-1):
        overhang = (index + local_range) - len(data)
        wrap_ind = overhang
        r_range = data[0:wrap_ind, :]
        r_range = np.concatenate((r_range, data[(index+1):(len(data)), :]))
        l_range = data[(index - local_range):(index), :]
    else:
        l_range = data[(index - local_range):(index), :]
        r_range = data[(index+1):(index+1 + local_range), :]
    return l_range, r_range



def hough_error(l_range, r_range, m_l, m_r, b_l, b_r):
    ''' Hough transform rrror range: run hough transform on pts above and below the hough line to get a box representing error 
    Input:
     - l_range, r_range: data left and right of local max
     - m_l, m_r, b_l, b_r: equations of Hough lines of R and L 
    Output: new Hough lines from datsets divided by the input Hough lines
    '''
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
    A_r_cart = [[0,0]]
    B_r_cart = [[0,0]]
    A_r_polar = [[0,0]]
    B_r_polar = [[0,0]]
    for k in range(len(r_range)):
        x_r = r_range[k][1]*math.cos(math.radians(r_range[k][0])) # x val = r*cos(theta)
        y_r = r_range[k][1]*math.sin(math.radians(r_range[k][0])) # y val = r*sin(theta)
        if((m_r*x_r + b_r) < y_r): # if for line x = x_r
            A_r_cart.append([x_r, y_r])
            A_r_polar.append([r_range[k][0], r_range[k][1]])
        else: 
            B_r_cart.append([x_r, y_r])
            B_r_polar.append([r_range[k][0], r_range[k][1]])
    A_r_cart = np.array(A_r_cart)
    B_r_cart = np.array(B_r_cart)
    A_l_cart = np.array(A_l_cart)
    B_l_cart = np.array(B_l_cart)
    A_r_polar = np.array(A_r_polar)
    B_r_polar = np.array(B_r_polar)
    A_l_polar = np.array(A_l_polar)
    B_l_polar = np.array(B_l_polar)
    A_l_theta, A_l_rho, A_l_acc_max = hough.hough_line(A_l_polar, 2)
    B_l_theta, B_l_rho, B_l_acc_max = hough.hough_line(B_l_polar, 2)
    A_r_theta, A_r_rho, A_r_acc_max = hough.hough_line(A_r_polar, 2)
    B_r_theta, B_r_rho, B_r_acc_max = hough.hough_line(B_r_polar, 2)
    m_A_r, b_A_r = hough_to_cartesian_line(A_r_rho, A_r_theta)
    m_B_r, b_B_r = hough_to_cartesian_line(B_r_rho, B_r_theta)
    m_A_l, b_A_l = hough_to_cartesian_line(A_l_rho, A_l_theta)
    m_B_l, b_B_l = hough_to_cartesian_line(B_l_rho, B_l_theta)

    # find 4 intersection points to make grid (1:[Au, Al], 2:[Au, Bl], 3:[Bu, Al], 4:[Bu, Bl])
    int_x_AuAl, int_y_AuAl = linear_intersection(m_A_r, b_A_r, m_A_l, b_A_l)
    int_x_AuBl, int_y_AuBl = linear_intersection(m_A_r, b_A_r, m_B_l, b_B_l)
    int_x_BuAl, int_y_BuAl = linear_intersection(m_B_r, b_B_r, m_A_l, b_A_l)
    int_x_BuBl, int_y_BuBl = linear_intersection(m_B_r, b_B_r, m_B_l, b_B_l)

    error_range = np.array([[int_x_AuAl, int_y_AuAl], [int_x_AuBl, int_y_AuBl], [int_x_BuAl, int_y_BuAl], [int_x_BuBl, int_y_BuBl]])
    return error_range, A_l_cart, A_r_cart, B_l_cart, B_r_cart, m_A_r, b_A_r, m_B_r, b_B_r, m_A_l, b_A_l, m_B_l, b_B_l


def plot_deviation(A_r_cart, B_r_cart, A_l_cart, B_l_cart, m_A_r, b_A_r, m_B_r, b_B_r, m_A_l, b_A_l, m_B_l, b_B_l):
    '''Plot the Hough deviation lines 
    '''
    plt.rcParams["figure.figsize"] = [7, 7]
    plt.rcParams["figure.autolayout"] = True

    # plt.scatter(A_r_cart[:, 0], A_r_cart[:, 1], c='red') 
    # plt.scatter(B_r_cart[:, 0], B_r_cart[:, 1], c='blue') 
    # plt.scatter(A_l_cart[:, 0], A_l_cart[:, 1], c='black') 
    # plt.scatter(B_l_cart[:, 0], B_l_cart[:, 1], c='green') 

    plt.scatter(A_r_cart[:, 0], A_r_cart[:, 1], c='red') 
    plt.plot([0, -b_A_r/m_A_r], [b_A_r, 0], color='red')
    
    plt.scatter(B_r_cart[:, 0], B_r_cart[:, 1], c='blue') 
    plt.plot([0, -b_B_r/m_B_r], [b_B_r, 0], color='blue')

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

def plot_maxima_search_area(l_range, r_range):
    ''' Plot Maxima Search Area (lower and upper)
    '''
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location("N")
    ax1.grid(True)
    for j in range(len(l_range)):
        ax1.plot(l_range[j][0]* -np.pi/180, l_range[j][1],'r+')
    for j in range(len(r_range)):
        ax1.plot(r_range[j][0]* -np.pi/180, r_range[j][1],'b+')    
    plt.show() 



def plot_search_and_hough(l_range, r_range, l_cartesian, r_cartesian, m_l, b_l, m_r, b_r):
    ''' Plot Maxima Search Area (lower and upper) & Hough Lines 
    '''
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location("N")
    ax1.grid(True)
    min_x_l = np.min(l_cartesian[:, 0])
    max_x_l = np.min(l_cartesian[:, 0])
    x_l = np.linspace(min_x_l-200, max_x_l+200, 100)
    y_l = m_l*x_l+b_l
    ax1.plot(-np.arctan2(y_l,x_l), np.sqrt(x_l**2+y_l**2), '-g', label='Lower Hough Line')
    min_x_r = np.min(r_cartesian[:, 0])
    max_x_r = np.min(r_cartesian[:, 0])
    x_r = np.linspace(min_x_r-200, max_x_r+200, 100)
    y_r = m_r*x_r+b_r
    ax1.plot(-np.arctan2(y_r,x_r), np.sqrt(x_r**2+y_r**2), '-y', label='Upper Hough Line')
    ax1.plot(-l_range[:, 0]* np.pi/180, l_range[:, 1],'r+')
    ax1.plot(-r_range[:, 0]* np.pi/180, r_range[:, 1],'b+')
    plt.show() 

def plot_data(data):
    ''' Plot Maxima Search Area (lower and upper) 
    '''
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location("N")
    ax1.grid(True)
    for j in range(len(data)):
        ax1.plot(data[j][0]* -np.pi/180, data[j][1],'r+')  
    plt.show() 




def match_transform(A, B):
    ''' Finds the translational and rotational matrices to transform points in A to points in B
         - make sure the cardinality of both sets A and B are the same
         - find the distance which is most different from the others
    Input:  
    - two sets of corners, A and B, which share some points altered by a transformation
    Return: 
    - dx: x-component of translation
    - dy: y-component of translation
    - dtheta: angle of rotation
    '''
    while(len(A) != len(B)): # make sure A is longer than B
        min_lst = []
        if len(A) < len(B):
            A, B = swap(A, B)
        for i in range(len(A)):
            min_dist = 10000000
            min_index = 0    
            for j in range(len(B)):
                print(A[i])
                dist = norm([A[i][0], A[i][1]], [B[j][0], B[j][1]])
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


def find_colinear_pt(m, x, y, d):
    ''' Finds the colinear point at distance d closest to the origin
    Input:
     - m: slope
     - x, y: point known to be on the line
     - d: disance from known point to colinear point
    Output:
     - x_c, y_c: the point colinear to the input line at distance d from (x, y)
    '''
    x_c = d/math.sqrt(1+(m*m))
    y_c = m*x_c
    
    x_c_p = x_c + x
    y_c_p = y_c + y
    d_p = norm([0, 0], [x_c_p, y_c_p])

    x_c_m = x - x_c
    y_c_m = y - y_c
    d_m = norm([0, 0], [x_c_m, y_c_m])

    if(d_m < d_p):
        x_c = x_c_m
        y_c = y_c_m
    else: 
        x_c = x_c_p
        y_c = y_c_p 
    
    return x_c, y_c




