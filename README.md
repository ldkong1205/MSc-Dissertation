# <center> 🎓 MSc Dissertation </center>

Start from `🕒13-Aug-2019`;     Last update at `🕤24-Feb-2020`. 
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
- Caijing Miao, Lingxi Xie, Fang Wan, Chi Su, Hongye Liu, Jianbin Jiao, Qixiang Ye, "SIXray:A Large-scale Security Inspection X-ray Benchmark for Prohibited Item Discovery in Overlapping Images". [[arXiv](https://arxiv.org/abs/1901.00303v1)]

- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, "You Only Look Once: Unified, Real-Time Object Detection". [[arXiv](https://arxiv.org/abs/1506.02640)]

- Joseph Redmon, Ali Farhadi, "YOLO9000: Better, Faster, Stronger". [[arXiv](https://arxiv.org/abs/1612.08242)]

- Joseph Redmon, Ali Farhadi, "YOLOv3: An Incremental Improvement". [[arXiv](https://arxiv.org/abs/1804.02767)]

- Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks". [[arXiv](https://arxiv.org/abs/1506.01497)]

Progress
-----

### I. YOLOv3

- Some tutorials about YOLOv3 can be found at [Tutorial 1](https://medium.com/@viirya/yolo-a-very-simple-tutorial-8d573a303480), [Tutorial 2](https://blog.csdn.net/guleileo/article/details/80581858), and [Tutorial 3](https://blog.csdn.net/m0_37192554/article/details/81092514).

- The installation and configuration of YOLOv3 have been completed and preliminary test has been carried out. 

- Details of the installation and configuration can be found at [YOLOv3](https://pjreddie.com/darknet/yolo/) and [YOLOv3 for Mac](https://bbs.csdn.net/topics/392556090?list=lz).
 
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

#### a. Objective
- Object detection and confidence evaluation with one stage (different from region proposal-based two-stage approaches which require selective search and regression).


#### b. Fundamental of CNN
- **Why CNN for image? (3 reasons)**
> **Property 1:** Some patterns are much smaller than the whole image. The neuron doesn't have to see the whole image to discover the pattern. Also, connecting to small region requires less parameters.

> **Property 2:** The same patterns appear in different regions.

> **Property 3:** Subsampling the pixels will not change the objects (patterns).

- **Convolution**
> Convolution v.s. Fully-Connected Network:

> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/cnn.png)|![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/cnn-v.s.-fullyconnected.png)|
> |---|---|

- **Max Pooling**

> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/pooling.png)|![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/maxpooling.png)|
> |---|---|


#### c. The Concept of YOLO
The YOLO's detection system (3 steps):

> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/YOLO-detection-system.png)|
> |---|

> **Step 1:** Resize the image to 448x448.

> **Step 2:** Run a single CNN on the image.

> **Step 3:** Threshold the resulting detections by the model's confidence.

The grid division:
> The system divides the image into an S x S grid. If the center of an object falls into a grid cell, that grid cell
is responsible for detecting that object.

> Each grid cell predicts B bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts.

> Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.

> Each bounding box consists of 5 predictions: x, y, w, h, and confidence. The (x, y) coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.

> Each grid cell also predicts C conditional class probabilities, Pr(Class_i|Object). These probabilities are conditioned on the grid cell containing an object.

> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/YOLO-model.png)|
> |---|








