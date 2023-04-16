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

    # Find the translation constants (x_trans, y_trans) and apply to the whole data set so the HT can be done in Q1
    x_trans = cart_arr.min(axis=0)[0]
    y_trans = cart_arr.min(axis=0)[1]
   
    for i in range(len(cart_arr)):
        if x_trans < 0:
            cart_arr[i][0] += -x_trans
        if y_trans < 0:
            cart_arr[i][1] += -y_trans  


    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(0.0, 180.0, .5))              # defaults to 50 samples from -90 to 90
    width = cart_arr.max(axis=0)[0]           # y range 
    height = cart_arr.max(axis=0)[1]            # x range
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
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
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, int(math.floor(t_idx/theta_bin_size))] += 1  #int(math.floor(rho/rho_bin_size))
    
    # --- Display Accumulator ---
    # # Scatter plot 
    # # !!!!!!!! Runtime Warning !!!!!!!!!
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

    # accumulator_data_lst = []
    # # print(accumulator.shape)
    # max_acc_val = 0
    # hough_theta = 0
    # hough_rho = 0
    # for i in range(accumulator.shape[0]): # rho
    #     for j in range(accumulator.shape[1]): # theta
    #         if (accumulator[i][j] > 0):
    #             rho_elt = i - diag_len          # mapping [0, 2*diag] (index) --> [-diag, diag] (real rho value)
    #             theta_elt = j                   # mapping [0, 360] (index) --> [0, 180] (real angle value)
    #             if (max_acc_val < accumulator[i][j]):
    #                 max_acc_val = accumulator[i][j]
    #                 hough_rho = rho_elt+min_y
    #                 hough_theta = math.radians(theta_elt)
    
    acc_max_val = accumulator.max()
    # non_zero = np.count_nonzero(accumulator)
    # total_val = np.product(accumulator.shape)
    # sparsity = (total_val - non_zero) / total_val
    # print(sparsity)

    # print(acc_max_val)


    idx = np.argmax(accumulator)                        # returns the index when searching row by row
    # print("lid", l_idx)
    # print(idx/accumulator.shape[1])
    # print("acc", accumulator.shape[1])
    if(len(rhos) == 0):
        rho = 0
    else: rho = rhos[round(idx / accumulator.shape[1])]
    # temp = idx % accumulator.shape[1]
    # print(l_accumulator.shape[1])
    theta = thetas[idx % accumulator.shape[1]]          # only theta neccessary to show orthogonality
    # return accumulator, thetas, rhos

    # Undo the translation in polar space
    theta = theta*bin_sz # theta stays the same 
    if x_trans < 0:
        rho -= -x_trans*math.cos(theta)
    if y_trans < 0:
        rho -= -y_trans*math.sin(theta)   

    return theta, rho, acc_max_val
