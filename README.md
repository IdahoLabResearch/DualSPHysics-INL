# DualSPHysics-INL
                            Screw conveyance                                      Cone penetrating in sands

<img src="https://github.com/user-attachments/assets/b1f556e4-17fb-49f8-927a-c5c88cde6b4a" alt="Screw" width="500" />
<img src="https://github.com/user-attachments/assets/8162c988-411b-4649-a872-1f53bc54d217" alt="CPT_animation" width="400" />

                                                      Biomass flow in hopper

<img src="https://github.com/user-attachments/assets/e405317e-1511-4718-b901-5a6ca8eb50bb" alt="30_animation" width="1000" />

                          Oedometer compression                               Vane Shearing
<img src="https://github.com/user-attachments/assets/3af6e29f-0e60-46c1-b129-5f8e91f3c828" alt="oed" width="500" />
<img src="https://github.com/user-attachments/assets/28fe9456-f0bc-445d-b1a7-46865e570d92" alt="vane" width="300" />

                            Static Angle of Repose

<img src="https://github.com/user-attachments/assets/e6f3197c-fb33-4e3f-8edf-76507b689a2b" alt="aor" width="600" />

**DualSPHysics-INL** is an extended adaptation of the [DualSPHysics](https://dual.sphysics.org/) open-source Smoothed Particle Hydrodynamics (SPH) simulation software based on DualSPHysics release version 5.0.1. Source code was modified from src_mphase/DSPH_v5.0_NNewtonian. 

**Why DualSPHysics-INL?** While DualSPHysics was developed for the simulation of fluid flow, DualSPHysics-INL was developed to simulate the flow of granular materials, such as soils and biomass materials. In DualSPHysics-INL, a critical state soil mechanics based G-B hypoplastic constitutive model was adopted that has the capability to simulate granular materials of a wide range of mechanical responses. The code adopts a momentum-based boundary condition that is able to sustain impact loading without particles leaking to the outside of the boundaries and can achieve a full range of frictional conditions, including free-slip and no-slip. The code adopts GPU-acceleration, enabling fast computation for complex problems.   

# Install DualSPHysics-INL
Since the code is based on DualSPHysics, most of the guidances for installation, pre- and post-processing and running from [DualSPHysics Wiki](https://github.com/DualSPHysics/DualSPHysics/wiki) still applies to DualSPHysics-INL. Please follow that Wiki link on how to install DualSPHysics-INL on either Linux, Windows or MacOS systems. Given that the code DualSPHysics-INL was developed and tested in Linux environment, a detailed installation guide using the Linux environment as an example will be provided below:

## Prerequesite
Not like the original DualSPHysics that supports both CPU and CPU&GPU computation, the current version of DualSPHysics-INL only supports CPU&GPU computation, thus, a cuda-GPU enabled computer or cluster is required to run the code.

### Check cuda-GPU compatibility
(to be added)
### Install Cuda
(to be added)

## Tested releases:
Linux Ubuntu 22.04 LTS (TO BE TESTED)
Linux Ubuntu 20.04 LTS + cuda-10.1 + gcc-9.4.0

## Download the LIGGGHTS-INL repository:
Option 1a. For users: `git clone https:xxx.git`

Option 1b. For users: `wget https://github.com/xxx.zip`

Option 2. For developers: `git clone git@github.com:xxx.git`

## Compiling the source code
Navigate to the repository folder DualSPHysics-INL/src-INL/source

`cd DualSPHysics-INL/src-INL/source`

Open Makefile (use such as emacs, nano, vim) and make modifications as needed:

`emacs Makefile`

Inside Makefile, if your gcc version is > 5.0, make sure `USE_GCC5=YES`.

For compiling to the release version, set `USE_DEBUG=NO`, for the debugging version, set it to YES and refer to the “[Developer](link)” section below.

Make sure `COMPILE_CHRONO`, `COMPILE_CHRONO_OMP`, `COMPILE_WAVEGEN` and `COMPILE_MOORDYN` are all set to NO since our current version of DualSPHysics-INL do not have the capability to couple with these functions.

Set the name and directory of the compiled file in `EXECNAME` and `EXECS_DIRECTORY` as desired.

In `"CUDA selection”` section, change `CUDAVER=114` to be compatible with the cuda version installed on your computer and change the corresponding `DIRTOOLKIT` and `"GPU architectures"`. 

After modifying the Makefile, save it and run (note if it's on cluster, use `module load` to load proper versions of gcc and cuda):

`make -j<number_of_thread> -f Makefile` (e.g. `make -j4 -f Makefile`)

If the compilation is successful, an executable with the name defined in Makefile `EXECNAME` should appear in the folder defined in `EXECS_DIRECTORY`.

## Quick start - run a test case
Navigate to the folder `DualSPHysics-INL/examples/test_AOR`. Make sure the input file `test_AOR_Def.xml` exist.

Run pre_processing.sh:

`bash pre_processing.sh`

A folder `test_AOR_out` should be generated, within which a file `test_AOR.xml` and a `test_AOR.bi4` should be generated, along with several vtk outputs that can be checked using Paraview or other similar software. 

If the pre-processing generates expected files, go ahead and run the main code:

`bash mainRunGPU.sh` (Note, in clusters, this command should be run in the batch mode, e.g. in INL sawtooth cluster, type `qsub mainRun_sawtoothGPU.pbs`, a template of the 'mainRun_sawtoothGPU.pbs' is attached in the folder but for a specific cluster, follows its user guide).

After the main simulation is completed, a series of data should be generated in a `data` folder, post-processing the data by:

`bash post_processing.sh`

You can also combine the pre-processing, main run and post-processing together by running:

`bash Run.sh`

However, it is always a good practice to run the pre-processing first and check if the initial status generated is correct before running the main simulation.

## DualSPHysics-INL documentation ##

### Capabilities ###

Granular flow (e.g. soils, biomass materials)

Output stresses, density, void ratio, velocity etc. of SPH nodes

Able to compute reaction forces and toruqe on boundaries

Able to assign free-slip, no-slip and Coulomb friction based frictional boundary conditions.

### Limitations ###

The original DualSPHysics code developed for fluid flow has multiple options such as coupling with other physical engines as described in [DualSPHysics Wiki](https://github.com/DualSPHysics/DualSPHysics/wiki). However, our current version of DualSPHysics-INL for granular flow has not adopted its full range of functions. The limitations are pointed below:
1. The code is only developed for CPU&GPU computing, a cuda GPU-enabled and properly configured computer must be used. Pure CPU compution was not developed.
DualSPHysics-INL does not allow couple with Chrono, MoorDyn+, Discrete Element Modeling (DEM) and wave propagation models.
2. The current code cannot be used for 2D simulations or with any symmetric boundary conditions. 
Only one granular phase is allowed at this point.
3. The floating scheme cannot be enabled. This means that a boundary can be assigned a fixed condition or designated motion and the reaction force from granular particles on the boundary can be computed, however, free motion of boundaries upon interaction with granular particles is not allowed.
### Theory ###
(to be added)

### Pre-Processing ###

### Post-processing ###

## Citing DualSPHysics-INL ##
Theory of the code / Static Angle of Repose of biomass materials / Oedometer compression of biomass materials
  Zhao, Y., Jin, W., Klinger, J., Dayton, D. C., & Dai, S. (2023). SPH modeling of biomass granular flow: Theoretical implementation and experimental validation. Powder Technology, 426, 118625. 

Other implementation used DualSPHysics-INL:

Biomass flow in hopper and auger

•	Zhao, Y., Ikbarieh, A., Jin, W., Klinger, J., Saha, N., Dayton, D. C., & Dai, S (2024). SPH modeling of biomass granular flow: engineering application in hopper and auger. ACS Sustainable Chemistry & Engineering. 12(10), 4213-4223.

A fully-connected Artificial Neuron Network (ANN) model for biomass hopper flow:

•	Ikbarieh, A., Jin, W., Zhao, Y., Saha, N., Klinger, J., Xia, Y., & Dai, S. Machine learning assisted cross-scale hopper design for flowing biomass granular materials (submitted to ACS Sustainable Chemistry & Engineering)

## License ##
(to be added)

## Developers ##


