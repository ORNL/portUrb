## Primary directory for compilation and running

From here, you'll be doing most of your compilation and simulation. There is a sample workflow on the main README.md of this repository demonstrating how this is typically done. Compilation is done through CMake.
* `cmakeclean.sh`: This executable script will clean your directory. If a `Makefile` is present, it'll run `make clean` and then remove the CMake files for a clean build to follow. Your output files will remain, but that should be about it.
* `cmakescript.sh`: This executable script will accept one parameter (the location of your experiments directory) and will build all dependencies as well as your experiment `.cpp` files for you to run later. It also links the `inputs` directory of your experiment folder to the current `build` directory so you have convenient access to your input files when running expeirments.
* `launchers`: This directory contains all of the HPC batch scripts for a few different machines as examples of how to launch your experiments.
* `machines`: This directory contains environment setup files for different HPC and laptop machines. In bash speak, you should `source` or `.` the appropriate file for your desired machine and configuration.
* `postproc`: This directory contains examples post processing files (mostly in python) for processing NetCDF output files.
