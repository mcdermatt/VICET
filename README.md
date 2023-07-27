# VICET
**V**elocity corrected **I**terative **C**ompact **E**llipsoidal **T**ransform

## Motion Distortion

A mechanically rotating LIDAR unit will produce a distorted representation of the scene if the sensor is in motion while recording. 
Consider the figure below which shows the different raw point clouds produced by static motion (a → a), forward linear motion (a → b), 
and composite translation and rotation (a → c) over the course of one scan period.   

![](https://github.com/mcdermatt/VICET/blob/main/wideFig1.jpg)

VICET is capable of jointly estimating both the rigid transform and the motion distortion compensation required to fit a distorted scan to a reference point cloud. 

<p float="left">
  <img src="/transOnlyBox.gif" width="400" />
  <img src="/transAndRotateBoxV2.gif" width="400" /> 
</p>

## Scan to HD Map Localization

![](https://github.com/mcdermatt/VICET/blob/main/scan2map1.gif)

Unlike existing techniques, VICET does not require a history of past motion to perform distortion correction.
This means that VICET is capable of accurately aligning a single higly distorted point cloud against an HD Map with no prior information.  
As we demonstrate in our paper, this allows VICET to achieve signficantly higher localization accuracy than rigid point cloud registration methods like NDT or ICP.  

## Bounding Error in LIDAR Odometry

Consider the case below where a new scene is being explored with a LIDAR sensor. Relative distortion between the two clouds makes it impossible to align all world features in both scans.
Without an a priori model of the enviornment, it is impossible to determine which of the two scans is actually representative of the real scene. 
