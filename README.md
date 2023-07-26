# VICET
$\textbf{V}$elocity corrected $textbf{I}$terative $textbf{C}$ompact $textbf{E}$llipsoidal $textbf{T}$ransform

## Motion Distortion

A mechanically rotating LIDAR unit will produce a distorted representation of the scene if the sensor is in motion while recording. 
Consider the figure below which shows the different raw point clouds produced by static motion (a → a), forward linear motion (a → b), 
and composite translation and rotation (a → c).   

![](https://github.com/mcdermatt/VICET/blob/main/wideFig1.jpg)

VICET is capable of jointly estimating the rigid transform and relative motion distortion states between two point clouds to accurately align complex surfaces. 

| ![](https://github.com/mcdermatt/VICET/blob/main/transOnlyBox.gif)  | ![](https://github.com/mcdermatt/VICET/blob/main/transAndRotateBoxV2.gif) |


## Scan to HD Map Localization

Unlike existing techniques, VICET does not require a history of past motion to perform distortion correction.
This means that VICET is capable of accurately aligning higly distorted point clouds against an HD Map. 

![](https://github.com/mcdermatt/VICET/blob/main/scan2map1.gif)
