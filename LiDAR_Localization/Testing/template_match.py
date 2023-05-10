
# from rplidar import RPLidar
# import hough_transform as hough

import sys

# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# import matplotlib.transforms as mtransforms
import numpy as np
import csv  
import math
# import cv2 as cv
import icp

'''
Experiment for verification of ICP 
'''

corners_I41_23 = [[1822.2525749182169, 1401.408975727723], 
                [1409.9422098508956, -2055.559826452765], 
                [-723.2683912112225, -3392.719482055958], 
                [-1714.6312359441654, -2461.1743053813743], 
                [-347.26273064566976, -483.54122019839315], 
                [-136.0215501403569, 986.0072470311222]]

corners_I41_22 = [[1462.686800890119, -2347.8623673887055], 
                [-1084.83939256345, 848.552141063747], 
                [-94.22709313507107, 681.8095716746273]]

corners_I41_21 = [[2130.9470481797302, 104.76902889911014], 
                [1202.4521955636244, -2796.783683078333], 
                [-644.2304275360875, -2917.770209644564], 
                [-530.2814157996413, -434.50532473542387], 
                [-982.0693293046612, 19.586170644601356], 
                [-991.9132463920835, 649.0871942137683], 
                [-20.461811589527997, 382.70147183317056]]

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

# This works when the chnage is small, but when there are large changes between timesteps

A = corners_I41_22
B = corners_I41_21
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

# https://github.com/ClayFlannigan/icp
T, R1, t1 = icp.best_fit_transform(np.array(A), np.array(B))
print("T:", T, "\nR1:", R1, "\nt1:", t1)

