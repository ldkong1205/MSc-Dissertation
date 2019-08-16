# 🎓 MSc Dissertation

Start from `🕒13-Aug-2019`;     Last update at `🕤16-Aug-2019`. 
<br>
  
Project Title
-----
Computational Imaging and Detection via Deep Learning
<br>

Project Summary
-----
Data-driven signal and data modeling has received much attention recently, for its promising performance in image processing, computer vision, imaging, etc. Among many machine learning techniques, the popular deep learning has demonstrated promising performance in image-related applications. However, it is still unclear whether it can be applied to benefit various computational imaging and vision applicartions, ranging from image restoration to analysis. This project aims to develop efficient and effective deep learning algorithms for computational imaging and detection applications.

Keywords: `Deep Learning`,  `Object Detection`,  `X-Ray Image`.
<br>

References
-----
- Caijing Miao, Lingxi Xie, Fang Wan, Chi Su, Hongye Liu, Jianbin Jiao, Qixiang Ye, "SIXray:A Large-scale Security Inspection X-ray Benchmark for Prohibited Item Discovery in Overlapping Images".

- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, "You Only Look Once: Unified, Real-Time Object Detection".

- Joseph Redmon, Ali Farhadi, "YOLO9000: Better, Faster, Stronger".

- Joseph Redmon, Ali Farhadi, "YOLOv3: An Incremental Improvement".

- Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".

Progress
-----

### I. YOLOv3

- Some tutorials about YOLOv3 can be found at https://medium.com/@viirya/yolo-a-very-simple-tutorial-8d573a303480, https://blog.csdn.net/guleileo/article/details/80581858, and https://blog.csdn.net/m0_37192554/article/details/81092514.

- The installation and configuration of YOLOv3 have been completed and preliminary test has been carried out. 

- Details of the installation and configuration can be found at https://pjreddie.com/darknet/yolo/ and https://bbs.csdn.net/topics/392556090?list=lz.
 
> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/predictions2.jpg)|![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/predictions%201.jpg)|![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/predictions.jpg)|
> |---|---|---|

- Important codes:
> Activating the detection for multiple images (use Ctrl-C to exit):
```python
./darknet detect cfg/yolov3.cfg yolov3.weights
```
> Changing the detection threshold:
```python
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0.25
```
<br>

### II. Some Notes about YOLO, YOLOv2, and YOLOv3

#### a. Aim
- object detection and confidence evaluation with one stage (different from region proposal-based two-stage approaches which require selective search and regression).

<br>

#### b. Fundamental of CNN
- **Why CNN for image? (3 reasons)**
> Property 1: Some patterns are much smaller than the whole image. The neuron doesn't have to see the whole image to discover the pattern. Also, connecting to small region requires less parameters.

> Property 2: The same patterns appear in different regions.

> Property 3: Subsampling the pixels will not change the objects (patterns).

> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/cnn.png)|
> |---|

- **Convolution**
> Convolution v.s. Fully-Connected Network:

> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/cnn-v.s.-fullyconnected.png)|
> |---|

- **Max Pooling**

> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/pooling.png)|
> |---|

<br>

#### c. The structure of YOLO

