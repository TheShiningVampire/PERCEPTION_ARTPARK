<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://github.com/TheShiningVampire/PERCEPTION-ARTPARK/blob/main/PERCEPTION_Logo.png"></a>
</p>

<h3 align="center">PERCEPTION FOR A SELF DRIVING CAR</h3>

<div align="center">
<!-- 
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE) -->

</div>

---

<p align="center"> This repository contains the codes written during the Summer Internship at ARTPARK, IISC Banglaore. The project is builing the PERCEPTION for a self driving car. Tasks such as Object detection, Multi-object tracking and Agent trajectory prediction are implemented.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Results](#results)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
Perception is a central problem for any autonomous
agent, be it humans, robots or self-driving vehicles. This
module helps for a smoother and more reliable control of
the car using the path-planning module of the autonomous
agent. It can also aid in pose estimation. For our project,
we have included the following sub-modules for the perception:
- Multi-object detection using the YOLOv5 algorithm
- Multi-object tracking using the Deep Sort algorithm
- Trajectory prediction using the PEC Net algorithm

The codes have been tried on Windows 10 and Windows 11 with Torch 1.9.0 and Cuda 11.1 on Nvidia Geforce RTX 3060 GPU.
## :trophy: Results obtained <a name = "results"></a>

The models were tried on Lyft level 5 dataset and the KITTI dataset.
The results obtained on the model are as follows:
## **Object Detection** 
- ### Lyft level 5 dataset

Original Video | YOLO v5 Predictions
--- | ---
![Lyft_level5_original](https://user-images.githubusercontent.com/55876739/132419488-98b0fc4c-8ecd-4b0e-9477-5cbe37f5c695.gif)  | ![Lyft_level5_detections](https://user-images.githubusercontent.com/55876739/132420202-de724efb-35b5-4d6e-9da0-344f9b73cea4.gif)


- ### KITTI Dataset

Original Video | YOLO v5 Predictions
--- | ---
![KITTI_original](https://user-images.githubusercontent.com/55876739/132419729-c7f44f0c-0fea-49a1-b465-82da58dca1f9.gif) | ![KITTI_detections](https://user-images.githubusercontent.com/55876739/132419750-6283628a-df3e-4465-8c70-4c10765ffa75.gif)

The FPS obtained after object detection are as follows:
<center>
<table>
  <tr>
    <td> </td>
    <td colspan="2">Lyft Level 5 Dataset</td>
    <td colspan="2">KITTI Dataset</td>
  </tr>
  <tr>
    <td> </td>
    <td>Avg FPS</td> <td> Min FPS</td>
    <td>Avg FPS</td> <td> Min FPS</td>
  </tr>
  <tr>
    <td>NVIDIA GeForce RTX 3060 mobile GPU</td>
    <td>22.19</td>
    <td>14.33</td>
    <td>25.86</td>
    <td>16.13</td>
  </tr>
  <tr>
    <td>NVIDIA Telsa T4 GPU</td>
    <td>30.15</td>
    <td>21.88</td>
    <td>31.35</td>
    <td>19.27</td>
  </tr>
</table>
</center>

## **Object Tracking** 

- ### Lyft level 5 dataset

Original Video | Deep Sort Tracking
--- | ---
![Scene_One](https://user-images.githubusercontent.com/55876739/134780076-04073d21-5cc2-4cab-a9ce-39923d9848fa.gif) | ![Scene_One_tracked](https://user-images.githubusercontent.com/55876739/134780081-0cd22132-ca90-42c9-b2c1-44c8c03883b1.gif)


- ### KITTI Dataset

Original Video | Deep Sort Tracking
--- | ---
![KITTI_original](https://user-images.githubusercontent.com/55876739/134780100-d585b125-10f1-43ea-8e75-1d6dce2a1527.gif) | ![KITTI_detected](https://user-images.githubusercontent.com/55876739/134780104-7302e390-8527-46d4-b10d-42e32eadc9ac.gif)

The FPS obtained after object detection are as follows:
<center>
<table>
  <tr>
    <td> </td>
    <td colspan="2">Lyft Level 5 Dataset</td>
    <td colspan="2">KITTI Dataset</td>
  </tr>
  <tr>
    <td> </td>
    <td>Avg FPS</td> <td> Min FPS</td>
    <td>Avg FPS</td> <td> Min FPS</td>
  </tr>
  <tr>
    <td>NVIDIA GeForce RTX 3060 mobile GPU</td>
    <td>12.96</td>
    <td>8.72</td>
    <td>13.14</td>
    <td>7.21</td>
  </tr>
  <tr>
    <td>NVIDIA Telsa T4 GPU</td>
    <td>14.19</td>
    <td>11.69</td>
    <td>14.04</td>
    <td>9.36</td>
  </tr>
</table>
</center>


## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## 🔧 Running the tests <a name = "tests"></a>

Explain how to run the automated tests for this system.

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## 🎈 Usage <a name="usage"></a>

Add notes about how to use the system.

## 🚀 Deployment <a name = "deployment"></a>

Add additional notes about how to deploy this on a live system.

## ⛏️ Built Using <a name = "built_using"></a>

- [MongoDB](https://www.mongodb.com/) - Database
- [Express](https://expressjs.com/) - Server Framework
- [VueJs](https://vuejs.org/) - Web Framework
- [NodeJs](https://nodejs.org/en/) - Server Environment

## ✍️ Authors <a name = "authors"></a>

- [@kylelobo](https://github.com/kylelobo) - Idea & Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References
