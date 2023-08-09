# VICET
**V**elocity corrected **I**terative **C**ompact **E**llipsoidal **T**ransform

## Motion Distortion

A mechanically rotating LIDAR unit will produce a distorted representation of the scene if the sensor is in motion while recording. 
Consider the figure below which shows the different raw point clouds produced by static motion (a → a), forward linear motion (a → b), 
and composite translation and rotation (a → c) over the course of one scan period.   

![](https://github.com/mcdermatt/VICET/blob/main/wideFig1.jpg)

VICET is capable of jointly estimating both the rigid transform and the relative motion correction required to fit a distorted scan to a reference point cloud. 
The "Rigid Transform" states represent the difference in pose between the origin of the keyframe scan and the origin of the new scan. 
The "Motion Correction" states estimated by VICET represent the apparent differences in the motion of the sensor during the recording of the second scan relative to the first. 
Correctly estimating these 12 parameters allows a distorted scan to be rectified to properly align with a keyframe scan.

<p float="left">
  <img src="/transOnlyBox.gif" width="400" />
  <img src="/transAndRotateBoxV2.gif" width="400" /> 
</p>

## Scan to HD Map Localization

![](https://github.com/mcdermatt/VICET/blob/main/scan2map1.gif)

One useful application of VICET is in the task of registering a raw distorted point cloud to an undistorted HD Map.   
Unlike existing techniques, VICET does not require a history of past motion or measurements from an external sensor to align the new point cloud.
As we demonstrate in our paper, this allows VICET to achieve signficantly higher localization accuracy than rigid point cloud registration methods like NDT or ICP.  

## Bounding Error in LIDAR Odometry

A second application of VICET is in aligning a distorted new LIDAR scan against a distorted reference cloud.
Consider the case below where a new scene is being explored with a LIDAR sensor. Relative distortion between the two clouds makes it impossible to align all world features in both scans.
Without an a priori model of the enviornment, it is unclear which of the two scans is actually representative of the real world as there is not enough information in the two scans alone to determine the true structure of the surrounding scene.

![](https://github.com/mcdermatt/VICET/blob/main/combinedDistortionMatchNoGround.jpg)

Relative motion distorion states between the two point clouds (obtained from VICET) can provide an upper bound on position error when registering the two scans.    
In short, this is because each component of the motion distortion vector represents how much the associated component in the rigid transfrom can "rattle around" when close to a correct solution.   

This uncertainty is demonstrated in the GIFs below. On the left, the blue scan is undistorted to match the red, and on the right the red cloud is undistorted to match the blue.
Note how the estimated distortion states between the two cases share simiar values at convergence, however with flipped signs.

<p float="left">
  <img src="/blueToRed.gif" width="400" />
  <img src="/redToBlue.gif" width="400" /> 
</p>
