# VICET
**V**elocity corrected **I**terative **C**ompact **E**llipsoidal **T**ransform

## Motion Distortion

A mechanically rotating LIDAR unit will produce a distorted representation of the scene if the sensor is in motion while recording. 
Consider the figure below which shows the different raw point clouds produced by static motion (a → a), forward linear motion (a → b), 
and composite translation and rotation (a → c).   

![](https://github.com/mcdermatt/VICET/blob/main/wideFig1.jpg)

VICET is capable of jointly estimating the rigid transform and relative motion distortion states between two point clouds to accurately align complex surfaces. 

<p float="left">
  <img src="/transOnlyBox.gif" width="400" />
  <img src="/transAndRotateBoxV2.gif" width="400" /> 
</p>

## Scan to HD Map Localization

![](https://github.com/mcdermatt/VICET/blob/main/scan2map1.gif)

Unlike existing techniques, VICET does not require a history of past motion to perform distortion correction.
This means that VICET is capable of accurately aligning a single higly distorted point cloud against an HD Map.  
