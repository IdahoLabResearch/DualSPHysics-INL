#!/bin/bash

fail () { 
 echo Execution aborted. 
 read -n1 -r -p "Press any key to continue..." key 
 exit 1 
}

# "name" and "dirout" are named according to the testcase

export name=Oedometer
export dirout=${name}_out
export diroutdata=${dirout}/data

# "executables" are renamed and called from their directory

export dirbin=../../bin/linux
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${dirbin}
export gencase="${dirbin}/GenCase_linux64"
export boundaryvtk="${dirbin}/BoundaryVTK_linux64"
export partvtk="${dirbin}/PartVTK_linux64"
export partvtkout="${dirbin}/PartVTKOut_linux64"

# Executes PartVTK to create VTK files with particles.
export dirout2=${dirout}/particles
${partvtk} -dirin ${diroutdata} -savevtk ${dirout2}/PartFluid -onlytype:-all,+fluid -vars:+rhop,+Sigma,+vel,+Void
${partvtk} -dirin ${diroutdata} -savevtk ${dirout2}/PartBound -onlytype:-all,+bound -vars:+mk,+Force
${boundaryvtk} -loadvtk ${dirout2}/PartBound_0010.vtk -motiondata ${diroutdata} -savevtkdata ${dirout2}/Bound
${partvtk} -dirin ${diroutdata} -savecsv ${dirout2}/PartBound -onlymk:13 -vars:+Force,-rhop,-type,-vel,-idp
${partvtkout} -dirin ${diroutdata} -savecsv ${dirout2}/PartFluidOut
if [ $? -ne 0 ] ; then fail; fi

if [ $option != 3 ];then
 echo All done
 else
 echo Execution aborted
fi

read -n1 -r -p "Press any key to continue..." key
