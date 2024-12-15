#!/bin/bash

fail () { 
 echo Execution aborted. 
 read -n1 -r -p "Press any key to continue..." key 
 exit 1 
}

# "name" and "dirout" are named according to the testcase

export name=hopper_32deg
export dirout=${name}_out
export diroutdata=${dirout}/data

# "executables" are renamed and called from their directory

export dirbin=../../bin/linux
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${dirbin}
export gencase="${dirbin}/GenCase_linux64"
export boundaryvtk="${dirbin}/BoundaryVTK_linux64"
export partvtk="${dirbin}/PartVTK_linux64"
export partvtkout="${dirbin}/PartVTKOut_linux64"

# Executes GenCase to create initial files for simulation.
${gencase} ${name}_Def ${dirout}/${name} -save:all
if [ $? -ne 0 ] ; then fail; fi

if [ $option != 3 ];then
 echo All done
 else
 echo Execution aborted
fi

read -n1 -r -p "Press any key to continue..." key