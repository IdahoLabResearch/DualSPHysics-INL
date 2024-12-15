import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
dirname = os.path.dirname(__file__)
import matplotlib.pyplot as plt

def compute_AoR(filename):
	# reader=vtk.vtkDataSetReader()
	# reader = vtk.vtkStructuredPointsReader()
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()
    n_points=data.GetNumberOfPoints()
	# point coorinates
    verts = vtk_to_numpy(data.GetPoints().GetData())

	# simga_ii = vtk_to_numpy(data.GetPointData().GetArray('Sigma'))
	# point_id = vtk_to_numpy(data.GetPointData().GetArray('Idp'))
    
    zQuater0 = np.min(verts[:,2])
    zQuater4 = np.max(verts[:,2])
    Height = zQuater4 - zQuater0
    zQuater1 = Height/4 + zQuater0
    zQuater2 = Height*2/4 + zQuater0
    zQuater3 = Height*3/4 + zQuater0
    ST = Height/20 # stripe thickness
    pos0 = np.empty([1,3])
    pos1 = np.empty([1,3])
    pos2 = np.empty([1,3])
    pos3 = np.empty([1,3])
    pos4 = np.empty([1,3])
    for n in range(0,len(verts)-1):
        if verts[n,2] <= (zQuater0 + ST):
            pos0 = np.concatenate((pos0,[[verts[n,0],verts[n,1],verts[n,2]]]),axis=0)
        if (verts[n,2] >= zQuater1 - ST/2) and (verts[n,2] <= (zQuater1 + ST/2)):
            pos1 = np.concatenate((pos1,[[verts[n,0],verts[n,1],verts[n,2]]]),axis=0)
        if (verts[n,2] >= zQuater2 - ST/2) and (verts[n,2] <= (zQuater2 + ST/2)):
            pos2 = np.concatenate((pos2,[[verts[n,0],verts[n,1],verts[n,2]]]),axis=0)
        if (verts[n,2] >= zQuater3 - ST/2) and (verts[n,2] <= (zQuater3 + ST/2)):
            pos3 = np.concatenate((pos3,[[verts[n,0],verts[n,1],verts[n,2]]]),axis=0)
        if verts[n,2] >= (zQuater4 - ST):
            pos4 = np.concatenate((pos4,[[verts[n,0],verts[n,1],verts[n,2]]]),axis=0)
    pos0 = np.delete(pos0,0,0)
    pos1 = np.delete(pos1,0,0)
    pos2 = np.delete(pos2,0,0)
    pos3 = np.delete(pos3,0,0)
    pos4 = np.delete(pos4,0,0)
    
    # Find max and min for 8 directions
    alpha = np.array([0,np.pi/8,2*np.pi/8,3*np.pi/8,4*np.pi/8,5*np.pi/8,6*np.pi/8,7*np.pi/8])
    ppos0 = np.empty([len(pos0),3]); # store projected positions
    ppos1 = np.empty([len(pos1),3]);
    ppos2 = np.empty([len(pos2),3]);
    ppos3 = np.empty([len(pos3),3]);
    ppos4 = np.empty([len(pos4),3]);
    nodes = np.empty([40,3]); # columns: height_index,orientation_index,diameter
    
    hi = 0 # height index
    oi = 0 # orientation index
    count = 0
    for orient in alpha:
        for n in range(0,len(pos0)):
            pv = pos0[n,0]*np.cos(orient) + pos0[n,1]*np.sin(orient)  
            ppos0[n,:] = [pv*np.cos(orient),pv*np.sin(orient),pos0[n,2]]
        nodes[count,0] = 0
        nodes[count,1] = oi
        nodes[count,2] = np.sqrt(np.power((np.max(ppos0[:,0]) - np.min(ppos0[:,0])),2) + np.power((np.max(ppos0[:,1]) - np.min(ppos0[:,1])),2)) 
        count = count + 1
        for n in range(0,len(pos1)):
            pv = pos1[n,0]*np.cos(orient) + pos1[n,1]*np.sin(orient)  
            ppos1[n,:] = [pv*np.cos(orient),pv*np.sin(orient),pos1[n,2]]
        nodes[count,0] = 1
        nodes[count,1] = oi
        nodes[count,2] = np.sqrt(np.power((np.max(ppos1[:,0]) - np.min(ppos1[:,0])),2) + np.power((np.max(ppos1[:,1]) - np.min(ppos1[:,1])),2)) 
        count = count + 1
        for n in range(0,len(pos2)):
            pv = pos2[n,0]*np.cos(orient) + pos2[n,1]*np.sin(orient)  
            ppos2[n,:] = [pv*np.cos(orient),pv*np.sin(orient),pos2[n,2]]
        nodes[count,0] = 2
        nodes[count,1] = oi
        nodes[count,2] = np.sqrt(np.power((np.max(ppos2[:,0]) - np.min(ppos2[:,0])),2) + np.power((np.max(ppos2[:,1]) - np.min(ppos2[:,1])),2)) 
        count = count + 1
        for n in range(0,len(pos3)):
            pv = pos3[n,0]*np.cos(orient) + pos3[n,1]*np.sin(orient)  
            ppos3[n,:] = [pv*np.cos(orient),pv*np.sin(orient),pos3[n,2]]
        nodes[count,0] = 3
        nodes[count,1] = oi
        nodes[count,2] = np.sqrt(np.power((np.max(ppos3[:,0]) - np.min(ppos3[:,0])),2) + np.power((np.max(ppos3[:,1]) - np.min(ppos3[:,1])),2)) 
        count = count + 1
        for n in range(0,len(pos4)):
            pv = pos4[n,0]*np.cos(orient) + pos4[n,1]*np.sin(orient)  
            ppos4[n,:] = [pv*np.cos(orient),pv*np.sin(orient),pos4[n,2]]
        nodes[count,0] = 4
        nodes[count,1] = oi
        nodes[count,2] = np.sqrt(np.power((np.max(ppos4[:,0]) - np.min(ppos4[:,0])),2) + np.power((np.max(ppos4[:,1]) - np.min(ppos4[:,1])),2)) 
        count = count + 1
        oi = oi + 1
    
    ave_radius0 = (nodes[0,2] + nodes[5,2] + nodes[10,2] + nodes[15,2] + nodes[20,2] + nodes[25,2] + nodes[30,2] + nodes[35,2])/8/2
    ave_radius1 = (nodes[1,2] + nodes[6,2] + nodes[11,2] + nodes[16,2] + nodes[21,2] + nodes[26,2] + nodes[31,2] + nodes[36,2])/8/2
    ave_radius2 = (nodes[2,2] + nodes[7,2] + nodes[12,2] + nodes[17,2] + nodes[22,2] + nodes[27,2] + nodes[32,2] + nodes[37,2])/8/2
    ave_radius3 = (nodes[3,2] + nodes[8,2] + nodes[13,2] + nodes[18,2] + nodes[23,2] + nodes[28,2] + nodes[33,2] + nodes[38,2])/8/2
    ave_radius4 = (nodes[4,2] + nodes[9,2] + nodes[14,2] + nodes[19,2] + nodes[24,2] + nodes[29,2] + nodes[34,2] + nodes[39,2])/8/2
    angle_of_repose = np.array([180/np.pi*np.arctan(Height/4/((ave_radius0 - ave_radius1))),180/np.pi*np.arctan(Height/4/((ave_radius1 - ave_radius2))),180/np.pi*np.arctan(Height/4/((ave_radius2 - ave_radius3))),180/np.pi*np.arctan(Height/4/((ave_radius3 - ave_radius4)))])
    # 4 AoR based on different height segement
    run_out_distance=ave_radius0
    return Height, run_out_distance, angle_of_repose

# file='Ratio21_1mmDF/PartFluid_0200.vtk'
# file='/home/jinw/projects/DualSPHysics/examples/mphase_nnewtonian/04_AoR/PartFluid_0136.vtk'

file = os.path.join(dirname, 'AoR_piner4_3.0_dp2_out/particles/PartFluid_0060.vtk')
h, r, AoR = compute_AoR(file)
print(h)
print(r)
print(AoR)

#ratio = np.array([1, 2, 4])

#plt.figure()
#plt.plot(ratio[0:2], AoR[0:2,0], 'go',linewidth=1.5, label = "DF 1mm")
#plt.plot(ratio[0:2], AoR[0:2,1], 'r*',linewidth=1.5, label = "DF 2mm")
#plt.plot(ratio[0:2], AoR[0:2,2], 'bs',linewidth=1.5, label = "DF 4mm")
#plt.legend(loc="upper right", prop={'size': 9})
#plt.xlabel("Aspect Ratio")
#plt.ylabel(r"AoR, $^o$")
#plt.tight_layout()
#plt.grid()
#plt.xlim(0, 4)
#plt.ylim(10, 35)
#plt.show()
#plt.savefig('AoR.png', facecolor='w', edgecolor='w',bbox_inches='tight')