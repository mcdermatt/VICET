# VICET
**V**elocity corrected **I**terative **C**ompact **E**llipsoidal **T**ransform

## Motion Distortion

A mechanically rotating LIDAR unit will produce a distorted representation of the scene if the sensor is in motion while recording. 
Consider the figure below which shows the different raw point clouds produced by static motion (a → a), forward linear motion (a → b), 
and composite translation and rotation (a → c) over the course of one scan period.   

![](https://github.com/mcdermatt/VICET/blob/main/gifs/wideFig1.jpg)

VICET is capable of jointly estimating both the rigid transform and the relative motion correction required to fit a distorted scan to a reference point cloud. 
The "Rigid Transform" states represent the difference in pose between the origin of the keyframe scan and the origin of the new scan. 
The "Motion Correction" states estimated by VICET represent the apparent differences in the motion of the sensor during the recording of the second scan relative to the first. 
Correctly estimating these 12 parameters allows a distorted scan to be rectified to properly align with a keyframe scan.

<p float="left">
  <img src="/gifs/transOnlyBox.gif" width="400" />
  <img src="/gifs/transAndRotateBoxV2.gif" width="400" /> 
</p>

## Scan to HD Map Localization

![](https://github.com/mcdermatt/VICET/blob/main/gifs/scan2map1.gif)

VICET is particularly useful in the task of registering a raw distorted point cloud to an undistorted HD Map.   
Motion of the sensor during the period of a LIDAR scan will stretch the resulting point cloud. Rigid scan registration techniques fail to account for this stretching and as a result will produce biased localization estimates. 
Strategies exist to account for motion distortion in point clouds, however they rely on either external sensor information or a sequence of multiple LIDAR scans to account for distortion. VICET is unique in that it can solve for both the rigid trnansform and motion distortion states required to properly align a distorted point cloud with a reference scan.
As we demonstrate in our paper, this allows VICET to achieve signficantly higher localization accuracy than rigid point cloud registration methods like NDT or ICP.  
</p>

## Cite VICET

Thank you for citing our work if you have used any of our code: 

[Correcting Motion Distortion for LIDAR HD-Map Localization](https://arxiv.org/pdf/2308.13694.pdf) 
```
@ARTICLE{10373094,
  author={McDermott, Matthew and Rife, Jason},
  journal={IEEE Robotics and Automation Letters}, 
  title={Correcting Motion Distortion for LIDAR Scan-to-Map Registration},
  year={2024},
  volume={9},
  number={2},
  pages={1516-1523},
  keywords={Laser radar;Distortion;Point cloud compression;Image sensors;Distortion measurement;Stators;Transforms;Localization;SLAM;range sensing},
  doi={10.1109/LRA.2023.3346757}
}

```
