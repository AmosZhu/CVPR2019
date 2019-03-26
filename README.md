# Depth from a polarisation + RGB stereo pair
![alt text](https://github.com/AmosZhu/CVPR2019/blob/master/Illustration.png)

This repostiory is the implementation of the paper. It contains two part: 1. Using graphical model to correct normal from polarisation images. 2. Estimate final depth with linear equation described by section 6 in the paper. These two parts constitute the pipeline of the paper, but they can run independently. e.g. If you have a coarse depthmap from other source (i.e kinect, multi-view), you can still run the code through the pipe line. Or you have a corrected normal map not from our "Normal correction" you can jump to "Depth estimation" but be specify with camera matrix.

If you have anything concerned, please fell free to contact me. 

# Normal correction
This part is implemented under python2.7 with OpenGM library, it takes a corse depth map (In theory, it can take any source of depthmap as long as it aligned with polarisation images. In our paper, the coarse depth is from stereo reconstruction) and polarisation images as input, The output is the corrected normal and estimated specular mask from polarisation information.

# Depth estimation
This part is implemented under Matlab. It takes corrected normal, (But can be any kind of "guide" surface normal), estimated specular mask, polarisation images, light source and camera matrix. It estimate the albedo and depth of the object.

# Tools

In order to install OpenGM please refer [Installing OpenGM with Python wrapper](https://memoryaux.wordpress.com/2014/08/15/installing-opengm-with-python-wrapper/), or by following command(Only support python2.7 and test on linux only)

      sudo apt-get install libopengm-bin libopengm-dev libopengm-doc python-opengm python-opengm-doc
      
 [Estimate polarisation information from images](https://github.com/waps101/depth-from-polarisation)
 

# Citation
If you use the code please cite:
