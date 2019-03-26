# Depth from a polarisation + RGB stereo pair

The implementation of the paper, this code contains two part: 1. Using graphical model to correct normal from polarisation images. 2. Estimate final depth with linear equation described by section 6 in the paper.

# Normal correction
This part is implemented under python2.7 with OpenGM library, it takes a corse depth map (In theory, it can take any source of depthmap. In our paper, the coarse depth is reconstructed from stereo) which aligned with the polarisation image. The output is the corrected normal from polarisation information.

