# Depth from a polarisation + RGB stereo pair
![alt text](http://url/to/img.png)
This repostiory is the implementation of the paper. The this code contains two part: 1. Using graphical model to correct normal from polarisation images. 2. Estimate final depth with linear equation described by section 6 in the paper. The corrected normal will be fed to the final depth estimation, but if you like, you can take 

# Get Coarse depth map
In the paper, we proposed 

# Normal correction
This part is implemented under python2.7 with OpenGM library, it takes a corse depth map (In theory, it can take any source of depthmap as long as it aligned with polarisation images. In our paper, the coarse depth is from stereo reconstruction) and polarisation images as input, The output is the corrected normal and estimated specular mask from polarisation information.

# Depth estimation
This part is implemented under Matlab. It takes corrected normal, (But can be any kind of "guide" surface normal), estimated specular mask, polarisation images, light source and camera matrix. It estimate the albedo and depth of the object.
