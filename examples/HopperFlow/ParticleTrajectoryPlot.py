# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:07:06 2023

@author: yzhao469
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import math

excel_file = "ParticleTrajectory.xlsx"
sheet_name = "80_0.77_0.251"
start_row = 11 #-1

# DF1mm_30
#end_rows = [212,177,195,227,144,98,119,54] # -2
#columns = ["E:G","H:J","K:M","N:P","W:Y","Z:AB","AC:AE","AL:AN"]

end_rows = [100,100,100,100,100,100,100,27] # -2
columns = ["B:D","E:G","H:J","K:M","N:P","Q:S","T:V","W:Y"]
#end_rows = [52,61,62,57,29,20,40,43] # -2
#columns = ["B:D","E:G","H:J","K:M","Q:S","W:Y","Z:AB","AC:AE"]

#end_rows = [27,61,90,89,96,97,83,81] # -2
#columns = ["B:D","H:J","N:P","Q:S","T:V","W:Y","Z:AB","AC:AE"]
#end_rows = [55,31,29,53,75,20,76,16] # -2
#columns = ["A:C","D:F","G:I","J:L","M:O","P:R","S:U","V:X"]

fig, ax = plt.subplots()
appendpoints = np.zeros((1,3))
for num in range(0,8):
    start_row = 10 # -1
    end_row = end_rows[num] 
    column = columns[num]
    columnnames = "EFG"
    data_frame = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=column,names=columnnames, skiprows=start_row, nrows=end_row - start_row + 1)
    x = data_frame["E"].tolist()
    y = data_frame["F"].tolist()
    z = data_frame["G"].tolist()
    v = []
    t = []
    for i in range(0, len(x)-1):
        v.append(math.sqrt((x[i+1]-x[i]) * (x[i+1]-x[i])+(y[i+1]-y[i]) * (y[i+1]-y[i])+(z[i+1]-z[i]) * (z[i+1]-z[i]))/0.06)
        t.append(0.06*i)
    v.append(0)
    t.append(t[len(t)-1]+0.06)
    
    # Create a figure and axis
    
    points = np.column_stack((x, z, v))
    points = np.column_stack((x,z,t))
    appendpoints = np.append(appendpoints,points,axis=0)
appendpoints = np.delete(appendpoints,0,0)
segments = [((appendpoints[i][0], appendpoints[i][1]), (appendpoints[i+1][0], appendpoints[i+1][1])) for i in range(len(appendpoints) - 1)]
#segments.pop(0)
for k in range(len(segments) - 1, -1, -1):
    if segments[k][1][1] > (segments[k][0][1] + 0.01):
        segments.pop(k)
cmap = plt.cm.get_cmap('RdBu')
rcmap=cmap.reversed()
lc = LineCollection(segments, cmap=rcmap, norm=plt.Normalize(0, 5.0))
cmap.reversed()
# Set line collection properties

for k in range(len(appendpoints) - 1, -1, -1):
    if appendpoints[k][2] == 0:
        appendpoints = np.delete(appendpoints,k,0)
        
lc.set_array(appendpoints[:,2])  # Assign color values from variable v
lc.set_linewidth(2)  # Set line width

endpoints = [segment[0] for segment in segments] + [segments[-1][1]]

# Convert the endpoints list into a NumPy array
endpoints = np.array(endpoints)

# Plot markers at the endpoints
ax.scatter(endpoints[:, 0], endpoints[:, 1], color='k', marker='o', s=10)

###
#linewidths = [min(max(0.5, 0.5 + 0.5 * v_i), 3.0) for v_i in appendpoints[:, 2]]
#lc.set_array(appendpoints[:, 2])  # Assign color values from variable v
#lc.set_linewidths(linewidths)

###
# Add the LineCollection to the axis
ax.add_collection(lc)
print("---")

# Set axis labels
font_dict = {'family': 'Arial', 'size': 18} 
ax.tick_params(axis='both', labelsize=16)
ax.set_xlabel(r'$\mathit{X}$ (m)', fontdict=font_dict)
ax.set_ylabel(r'$\mathit{Z}$ (m)', fontdict=font_dict)
ax.set_xlim(-0.3,0.3)
ax.set_ylim(0,0.55)

# Add colorbar
cbar = plt.colorbar(lc, ax=ax)
#cbar.set_label('Velocity [m/s]')  # Change 'Color Label' to your desired label
cbar.set_label('Time (s)')

#segments_wall1= [([-0.282,0.444,],[-0.014,0.020])] #30mm
#segments_wall2= [([0.282,0.444,],[0.014,0.020])]
segments_wall1= [([-0.282,0.444,],[-0.034,0.020])] #70mm
segments_wall2= [([0.282,0.444,],[0.034,0.020])]
segments_wall1= [([-0.282,0.444,],[-0.039,0.020])] #80mm
segments_wall2= [([0.282,0.444,],[0.0339,0.020])]

# Create a LineCollection for the additional line segments
lc_wall1 = LineCollection(segments_wall1, color='black', linewidth=1.5)
ax.add_collection(lc_wall1)
lc_wall2 = LineCollection(segments_wall2, color='black', linewidth=1.5)
ax.add_collection(lc_wall2)

plt.show()