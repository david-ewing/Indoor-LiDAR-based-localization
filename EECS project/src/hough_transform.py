import numpy as np
import math
from matplotlib import pyplot as plt

# code based on https://alyssaq.github.io/2014/understanding-hough-transform/

# Input: 
#   arr -  an array where the first element is theta and second is range
#   bin_sz - the bin size for theta values in the accumulator
def hough_line(arr, bin_sz):

    # convert to cartesian coords
    cart_arr = np.zeros([len(arr), 2]) # L by 2 np array
    for i in range(len(arr)):
        theta = arr[i][0]
        r = arr[i][1]
        cart_arr[i][0] = r*math.cos(math.radians(theta)) # x val
        cart_arr[i][1] = r*math.sin(math.radians(theta)) # y val
    # print(cart_arr)

    # Rho and Theta ranges
    # *************how do you deal with negative values when calculating the width and height*******
    thetas = np.deg2rad(np.arange(0.0, 180.0, .5))              # defaults to 50 samples from -90 to 90
    min_x = cart_arr.min(axis=0)[0]
    min_y = cart_arr.min(axis=0)[1]
    max_x = cart_arr.max(axis=0)[0]
    max_y = cart_arr.max(axis=0)[1]
    min_dist = math.sqrt(min_y ** 2 + min_x ** 2)
    # print(min_x, min_y)
    width = abs(max_y - min_y)           # y range 
    height = abs(max_x - min_x)            # x range
    # print("maximims: (", max_x, max_y, ")")
    # print("minimums: (", min_x, min_y, ")")
    # print(width, height)
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    # print("diag", diag_len)
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)     # returns evenly spaced valued from +/- max_dist

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    theta_bin_size = bin_sz # theta bin size of 2 seems to work best
    rho_bin_size = 1
    accumulator = np.zeros((int(math.ceil((2 * diag_len)/rho_bin_size))+1, int(math.floor(num_thetas/theta_bin_size))), dtype=np.uint64)
    
    # print(np.shape(accumulator))
    x_idxs = cart_arr[:, 0]
    y_idxs = cart_arr[:, 1]

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        # print(i, len(x_idxs))
        # address coordinates in each quadrant
        # if (max_x < 0 and max_y < 0):
        #     x = -(x_idxs[i] - min_x)
        #     y = -(y_idxs[i] - min_y)
        if(max_x < 0 and max_y >= 0):
            x = -(x_idxs[i] - min_x)
            y = (y_idxs[i] - min_y)
        elif(max_y < 0 and max_x >= 0):
            x = -(x_idxs[i] - min_x)
            y = (y_idxs[i] - min_y)
        else:
            x = x_idxs[i] - min_x
            y = y_idxs[i] - min_y

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, int(math.floor(t_idx/theta_bin_size))] += 1  #int(math.floor(rho/rho_bin_size))
    
    # --- Display Accumulator ---
    # Scatter plot 
    # !!!!!!!! Runtime Warning !!!!!!!!!
    # accumulator_data_lst = []
    # print(accumulator.shape)
    # for i in range(accumulator.shape[0]): # rho
    #     for j in range(accumulator.shape[1]): # theta
    #         if (accumulator[i][j] > 0):
    #             rho_elt = i - diag_len          # mapping [0, 2*diag] (index) --> [-diag, diag] (real rho value)
    #             theta_elt = j    # mapping [0, 360] (index) --> [0, 180] (real angle value)
    #             acc_value = accumulator[i][j]
    #             accumulator_data_lst.append([theta_elt, rho_elt, acc_value])
    # accumulator_data = np.array(accumulator_data_lst)
    
    acc_max_val = accumulator.max()
    print(acc_max_val)
    # plt.rcParams["figure.figsize"] = [7, 7]
    # plt.rcParams["figure.autolayout"] = True
    # plt.scatter(accumulator_data[:, 0], accumulator_data[:, 1]+min_y, c=accumulator_data[:, 2], s=1) 
    # plt.xlabel("Theta (degrees)")
    # plt.ylabel("Rho (mm)")
    # plt.title("Hough Accumulator")
    # # Display the plot
    # plt.show() 

    # # Image display (with intensity)
    # # Set the figure size
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    # # Random data points
    # data = accumulator
    # # Plot the data using imshow with gray colormap
    # plt.imshow(data, cmap='gray')
    # plt.xlabel("Theta (deg/2 -90)")
    # plt.ylabel("Rho (mm)")
    # plt.title("Hough Accumulator")
    # # Display the plot
    # plt.show()

    # np.savetxt("accumulator", accumulator)

    idx = np.argmax(accumulator)                        # returns the index when searching row by row
    # print("lid", l_idx)
    # print(idx/accumulator.shape[1])
    # print("acc", accumulator.shape[1])
    rho = rhos[idx % accumulator.shape[1]]
    print(rho)
    if(rho > 0):
        rho = rho - min_dist
    else: rho = rho + min_dist
    temp = idx % accumulator.shape[1]
    # print(l_accumulator.shape[1])
    theta = thetas[idx % accumulator.shape[1]]          # only theta neccessary to show orthogonality
    # return accumulator, thetas, rhos
    return theta*bin_sz, rho, acc_max_val