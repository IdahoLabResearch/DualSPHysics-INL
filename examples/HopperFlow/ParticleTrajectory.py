# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:59:02 2023

@author: yzhao469
"""
import vtk
import numpy as np

#################
MaxFrame = 100
dt = 0.06 # [s]
IdpValue = 355853# select the point of interest
#################

reader = vtk.vtkDataSetReader()
'''
point_pos= np.zeros([MaxFrame+1,3])
for i in range(MaxFrame+1):
    filename = f"PartFluid_{i:04d}.vtk"
    reader.SetFileName("hopper_32deg_DF_1mm_70_noslip_out/particles/"+ filename)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()


    point_data = data.GetPointData()
    Idp = point_data.GetArray("Idp")
    for ID in range(Idp.GetNumberOfValues()):
        if IdpValue == Idp.GetValue(ID):
            point = points.GetPoint(ID)
            pointnp = np.array(point)
            point_pos[i,0] = pointnp[0]
            point_pos[i,1] = pointnp[1]
            point_pos[i,2] = pointnp[2]
'''
point_pos = np.zeros((MaxFrame + 1, 3))
for i in range(MaxFrame + 1):
    filename = f"PartFluid_{i:04d}.vtk"
    reader.SetFileName("hopper_32deg_DF_1mm_70_noslip_out/particles/" + filename)
    reader.Update()
    data = reader.GetOutput()
    point_data = data.GetPointData()
    Idp = point_data.GetArray("Idp")
    
# Find the index of the point with IdpValue if it exists
    indices = np.where(Idp == IdpValue)[0]
    
    if indices.size > 0:
        # Get the first occurrence of IdpValue
        ID = indices[0]
        point = data.GetPoints().GetPoint(ID)
        point_pos[i] = np.array(point)
    else:
        # Handle the case when IdpValue is not found, e.g., set point_pos[i] to a default value.
        point_pos[i] = np.array([0.0, 0.0, 0.0])
