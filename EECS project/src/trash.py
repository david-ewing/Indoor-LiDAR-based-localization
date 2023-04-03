# wrap polar corrdinates to find the approproate range in non-end-looped list
if ((index - maxima_inspection_range) < 0): 
    overhang = -(index - maxima_inspection_range)
    wrap_ind = data_size-1 - overhang
    l_range = lidar_data[0:(index-1), :]
    l_range = np.concatenate((l_range, lidar_data[wrap_ind:(data_size-1), :]))
    u_range = lidar_data[(index+1):(index + maxima_inspection_range), :]
elif((index + maxima_inspection_range) > data_size-1):
    overhang = (index + maxima_inspection_range) - data_size-1
    wrap_ind = overhang
    u_range = lidar_data[0:wrap_ind, :]
    u_range = np.concatenate((u_range, lidar_data[(index+1):(data_size-1), :]))
    l_range = lidar_data[(index - maxima_inspection_range):(index-1), :]
else:
    l_range = lidar_data[(index - maxima_inspection_range):(index-1), :]
    u_range = lidar_data[(index+1):(index + maxima_inspection_range), :]
    

l_min = l_range[0][1] 
l_max = l_range[0][1]
u_min = u_range[0][1]
u_max = u_range[0][1]
for k in range(len(l_range)):
    if l_min > l_range[k][1]:
        l_min = l_range[k][1]
    if l_max < l_range[k][1]:
        l_max = l_range[k][1]
for k in range(len(u_range)):
    if u_min > u_range[k][1]:
        u_min = u_range[k][1]
    if u_max < u_range[k][1]:
        u_max = u_range[k][1]


if(x_ < 0 and y_ > 0): #Q1 (counterclockwise)
    theta = -(90-(np.arctan2(y_, x_)*180/math.pi))
elif(x_ < 0 and y_ < 0): #Q2 (cc)
    theta = (np.arctan2(y_, x_)*180/math.pi)+90
elif(x_ > 0 and y_ < 0): #Q3 (cc)
    theta = (90-(np.arctan2(y_, x_)*180/math.pi))+180
else: #Q4 (cc)
    theta = (np.arctan2(y_, x_)*180/math.pi)+270