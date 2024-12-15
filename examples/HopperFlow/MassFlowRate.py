# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:59:45 2023

@author: yzhao469
"""

# -*- coding: utf-8 -*-
# importing module
import glob

########################################################
def CalMassFlowRate(Path,MassFluid,PartTimeInterval,widthadj):
    #path: run the py code at the same level with case_out folder
    filename = glob.glob(Path + "/Run.out")
    start_line = "Part_0001"
    end_line = "Simulation"
    data = []
    with open(filename[0], 'r') as file:
        # read the lines of the file until the line containing "===" is found
        for line in file:
            if start_line in line:
                data.append(line)
                break  # exit the loop after the start line is found
        for line in file:
            if end_line in line:
                data.append(line)
                break  # exit the loop after the end line is found
            else:
                data.append(line)
    file.close()           
    data.remove(line)

    for i in range(len(data)):
        data[i] = data[i].split(' ')
    Time = []
    PartTime = []
    PartOut = []
    MassFlowRate = []
    MassFlowed = [0]
    k = 0
    for i in range(len(data)-2):
        if data[i][0].startswith("Part_"):
            if data[i][5] == '' and data[i][6] != '':
                Time.append(float(data[i][6]))
            if data[i][6] == '' and data[i][5] != '':
                Time.append(float(data[i][5]))
            if data[i][4] != '' and data[i][5] == '':
                Time.append(float(data[i][4]))
            PartTime.append(data[i][0])
            if data[i+1][3].startswith("out"):
                PartOut.append(int(data[i+1][4]))
            else:
                PartOut.append(0)
            MassFlowRate.append(PartOut[k]*MassFluid/PartTimeInterval/1000*3600/widthadj)
            if i > 0:
                MassFlowed.append(MassFlowed[k-1] + MassFlowRate[k]*PartTimeInterval/3600)
            k += 1

    return Time, PartTime, PartOut, MassFlowRate, MassFlowed

def main():
    path = "temp_out"
    Massfluid = 0.00000154152 # obtian from the Run.out file
    PartTimeInterval = 0.06 # [s], obtian from the Run.out file
    WidthAdj = 1/4 # in simulation, the width is set only 1/4 of the experiment
    [Time,PartTime,PartOut,MassFlowRate,MassFlowed] = CalMassFlowRate(path,Massfluid,PartTimeInterval,WidthAdj)
                            # [ton/h]     ton
    print("All done! ---")
    
if __name__ == "__main__":
    main()
