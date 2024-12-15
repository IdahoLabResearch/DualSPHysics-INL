# -*- coding: utf-8 -*-
# importing module
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse

####################### Arguments ######################
parser = argparse.ArgumentParser(description = 'DualSPHysics_Bound_ForceTorque calculation')
parser.add_argument('-path', nargs = "?") # e.g. "SlidingBlock_small_out"
parser.add_argument('-ForceCal', nargs = "?",type = bool)
parser.add_argument('-TorqueCal', nargs='?',type = bool)
parser.add_argument('-Mk', nargs='?', type = int) #11
parser.add_argument('-AxisP1',nargs='+',type=float)
parser.add_argument('-AxisP2',nargs='+',type=float) # 0 0 0.045; 0 1 0.045
args = parser.parse_args()
path = args.path
ForceCal = args.ForceCal
TorqueCal = args.TorqueCal
Mk = args.Mk
AxisP1 = args.AxisP1
AxisP2 = args.AxisP2
#path = 'ModifiedAuger_0015'
#ForceCal = args.ForceCal
#TorqueCal = args.TorqueCal
#Mk = args.Mk
#AxisP1 = args.AxisP1
#AxisP2 = args.AxisP2
########################################################
def InputData(path, Mk_assigned):
    #path: run the py code at the same level with case_out folder
    #Mk_assigned: decide the Mk of boundary particles to consider
    files = glob.glob(path + "/particles/PartBound*.csv")

    PosX = [];    PosY = [];    PosZ = [];
    ForceX = [];    ForceY = [];    ForceZ = [];
    #Mkk = [];     Mk = [];

    for filename in files:
         if filename == path+"/particles\\PartBound_stats.csv":
             continue
         df = pd.read_csv(filename, sep=";", header=2)
         # df: Index Pos.x[m] Pos.y[m] Pos.z[m] Idp Type Mk Force.x Force.y Force.z
         Posx = df['Pos.x [m]'].values.tolist()
         Posy = df['Pos.y [m]'].values.tolist()
         Posz = df['Pos.z [m]'].values.tolist()
         Forcex = df['Force.x'].values.tolist()
         Forcey = df['Force.y'].values.tolist()
         Forcez = df['Force.z'].values.tolist()
         #Mkk = df['Mk'].values.tolist()
         PosX.append(Posx)
         PosY.append(Posy)
         PosZ.append(Posz)
         ForceX.append(Forcex)
         ForceY.append(Forcey)
         ForceZ.append(Forcez)
         #Mk.append(Mkk)
         
    Pos_x = np.zeros((np.size(PosX,0),np.size(PosX,1)));
    Pos_y = np.zeros((np.size(PosY,0),np.size(PosY,1))); 
    Pos_z = np.zeros((np.size(PosZ,0),np.size(PosZ,1))); 
    Force_x = np.zeros((np.size(ForceX,0),np.size(ForceX,1))); 
    Force_y = np.zeros((np.size(ForceY,0),np.size(ForceY,1))); 
    Force_z = np.zeros((np.size(ForceZ,0),np.size(ForceZ,1))); 

    for i in range(0,np.size(PosX,0)):
         for j in range(0,np.size(PosX,1)):
             #if Mk[i][j] == Mk_assigned:
             Pos_x[i][j] = PosX[i][j]
             Pos_y[i][j] = PosY[i][j]
             Pos_z[i][j] = PosZ[i][j]
             Force_x[i][j] = ForceX[i][j]
             Force_y[i][j] = ForceY[i][j]
             Force_z[i][j] = ForceZ[i][j]
    
    return Pos_x, Pos_y, Pos_z, Force_x, Force_y, Force_z

def CalculateNetForce(Pos_x, Pos_y, Pos_z, Force_x, Force_y, Force_z):
    Netf = np.zeros((1,3)); 
    forcecenter = np.zeros((1,3));
    NetFx = np.zeros((np.size(Pos_x,0),1)); 
    NetFy = np.zeros((np.size(Pos_y,0),1));
    NetFz = np.zeros((np.size(Pos_z,0),1));
    ForceCenterx = np.zeros((np.size(Pos_x,0),1));
    ForceCentery = np.zeros((np.size(Pos_y,0),1));
    ForceCenterz = np.zeros((np.size(Pos_z,0),1));
    # Force acting on a boundary particle f = acceleration * Boundary mass
    # Check boundary mass from the run.out file    
  
    for i in range(0,np.size(Force_x,0)): #time steps
        FPx_sum = 0; FPy_sum = 0; FPz_sum = 0;
        NetFx_sum = 0; NetFy_sum = 0; NetFz_sum = 0;
        for j in range(0,np.size(Force_x,1)): #number of bound particles
            FPx_sum = FPx_sum + Force_x[i][j]*Pos_x[i][j]
            NetFx_sum = NetFx_sum + Force_x[i][j]
            FPy_sum = FPy_sum + Force_y[i][j]*Pos_y[i][j]
            NetFy_sum = NetFy_sum + Force_y[i][j]
            FPz_sum = FPz_sum + Force_z[i][j]*Pos_z[i][j]
            NetFz_sum = NetFz_sum + Force_z[i][j]
        Netf = [NetFx_sum,NetFy_sum,NetFz_sum]
        forcecenter = [0,0,0]
        if NetFx_sum == 0 and NetFy_sum !=0 and NetFz_sum != 0:
            print('Sum of force in x direction is zero!')
            forcecenter = [9999.0,FPy_sum/NetFy_sum,FPz_sum/NetFz_sum]
        elif NetFx_sum != 0 and NetFy_sum ==0 and NetFz_sum != 0:
            print('Sum of force in y direction is zero!')
            forcecenter = [FPx_sum/NetFx_sum,9999.0,FPz_sum/NetFz_sum]
        elif NetFx_sum != 0 and NetFy_sum !=0 and NetFz_sum == 0:
            print('Sum of force in z direction is zero!')
            forcecenter = [FPx_sum/NetFx_sum,FPy_sum/NetFy_sum,9999.0]           
        elif NetFx_sum == 0 and NetFy_sum ==0 and NetFz_sum != 0:
            print('Sum of force in x and y direction are zero!')
            forcecenter = [9999.0,9999.0,FPz_sum/NetFz_sum]            
        elif NetFx_sum == 0 and NetFy_sum !=0 and NetFz_sum == 0:
            print('Sum of force in x and z direction are zero!')
            forcecenter = [9999.0,FPy_sum/NetFy_sum,9999.0]           
        elif NetFx_sum != 0 and NetFy_sum ==0 and NetFz_sum == 0:
            print('Sum of force in y and z direction are zero!')
            forcecenter = [FPx_sum/NetFx_sum,9999.0,9999.0]               
        elif NetFx_sum == 0 and NetFy_sum ==0 and NetFz_sum == 0:
            print('Sum of force in all three directions are zero!')
        else:
            forcecenter = [FPx_sum/NetFx_sum,FPy_sum/NetFy_sum,FPz_sum/NetFz_sum]

        NetFx[i] = Netf[0]
        NetFy[i] = Netf[1]
        NetFz[i] = Netf[2]
        ForceCenterx[i] = forcecenter[0]
        ForceCentery[i] = forcecenter[1]
        ForceCenterz[i] = forcecenter[2]
        
    # Plot net force over time
    #plt.plot(np.arange(0,np.size(fx,0),dtype=float),NetFx,'r-', \
    #         np.arange(0,np.size(fy,0),dtype=float),NetFy,'g-', \
    #         np.arange(0,np.size(fz,0),dtype=float),NetFz,'b-')
    #plt.title("Net force")
    #plt.xlabel("Output time step")
    #plt.ylabel("Forces")
    #plt.legend(['Fx','Fy','Fz'])
    #plt.show()
    
        
    return NetFx, NetFy, NetFz, ForceCenterx, ForceCentery, ForceCenterz
    
def CalculateTorque(Pos_x, Pos_y, Pos_z, Force_x, Force_y, Force_z,AxisP1,AxisP2):
    # Using two points on the axis to define axis vector
    AxisP1 = np.array(AxisP1)
    AxisP2 = np.array(AxisP2)
    v = AxisP2 - AxisP1
    v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2]
    Torque = np.zeros((np.size(Pos_x,0),1)); 
    
    for i in range(0,np.size(Pos_x,0)): #time steps
        T_sum = [0,0,0]    
        for j in range(0,np.size(Pos_x,1)):#number of bound particles
            # r vector: pointing to the boundary particle P from its orthogonal projected point on axis v
            rx = Pos_x[i][j] - 1/v2 * (v[0]*v[0]*Pos_x[i][j] + v[0]*v[1]*Pos_y[i][j] + v[0]*v[2]*Pos_z[i][j]) 
            ry = Pos_y[i][j] - 1/v2 * (v[1]*v[0]*Pos_x[i][j] + v[1]*v[1]*Pos_y[i][j] + v[1]*v[2]*Pos_z[i][j]) 
            rz = Pos_z[i][j] - 1/v2 * (v[2]*v[0]*Pos_x[i][j] + v[2]*v[1]*Pos_y[i][j] + v[2]*v[2]*Pos_z[i][j]) 
            #tau = v x r; the direction of the force that contributes to the torque at point P
            taux = v[1]*rz - v[2]*ry
            tauy = v[2]*rx - v[0]*rz
            tauz = v[0]*ry - rx*v[1]
            tau2 = taux*taux + tauy*tauy + tauz*tauz
            if tau2 != 0:
                ftimetau = Force_x[i][j]*taux + Force_y[i][j]*tauy + Force_z[i][j]*tauz
                # Project force f at point P onto the tau direction
                fstarx = ftimetau/tau2*taux
                fstary = ftimetau/tau2*tauy
                fstarz = ftimetau/tau2*tauz
                # T = r x fstar; Compute the torque given the r vector and the projected f
                Tx = ry*fstarz - rz*fstary
                Ty = rz*fstarx - rx*fstarz
                Tz = rx*fstary - ry*fstarx
            else:
                Tx = 0; Ty = 0; Tz = 0;
            T_sum = [T_sum[0]+Tx,T_sum[1]+Ty,T_sum[2]+Tz]
        # Torque direction is the same with v. Instead of given the coordinate components, here gives the absolute value
        Torque[i] = (T_sum[0]*v[0] + T_sum[1]*v[1] + T_sum[2]*v[2])/v2 
    
    return Torque
    
def main():
    [Pos_x, Pos_y, Pos_z, Force_x, Force_y, Force_z] = InputData(path,Mk)
    print('Data file is:',path)
    # If compute net force
    if ForceCal == True:
        NetFx, NetFy, NetFz, ForceCenterx, ForceCentery, ForceCenterz = CalculateNetForce(Pos_x, Pos_y, Pos_z, Force_x, Force_y, Force_z)
        con = np.concatenate((NetFx, NetFy, NetFz, ForceCenterx, ForceCentery, ForceCenterz),axis=1)
        NetF_file = open(path + "/particles/NetForceResult.csv",'w',newline='')
        header = ['NetFx N]','NetFy N]','NetFz N]','ForceCenterx [m]','ForceCentery [m]','ForceCenterz [m]']
        writer = csv.writer(NetF_file)
        writer.writerow(header)
        writer.writerows(con)
        NetF_file.close()
    # If compute torque along an axis
    if TorqueCal == True:
        Torque = CalculateTorque(Pos_x, Pos_y, Pos_z, Force_x, Force_y, Force_z,AxisP1,AxisP2)
        torque_file = open(path + "/particles/TorqueResult.csv",'w',newline='')
        writer = csv.writer(torque_file)
        header = ['Torque [N*m]']
        writer.writerow(header)
        writer.writerows(Torque)
        torque_file.close()

    print("All done! ---")
    
if __name__ == "__main__":
    main()

    



