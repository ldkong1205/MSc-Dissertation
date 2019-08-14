# 🎓 MSc Dissertation

Start from `🕒13-Aug-2019`;     Last update at `🕤14-Aug-2019`. 
<br>
  
Project Title
-----
Computational Imaging and Detection via Deep Learning
<br>

Project Summary
-----
Data-driven signal and data modeling has received much attention recently, for its promising performance in image processing, computer vision, imaging, etc. Among many machine learning techniques, the popular deep learning has demonstrated promising performance in image-related applications. However, it is still unclear whether it can be applied to benefit various computational imaging and vision applicartions, ranging from image restoration to analysis. This project aims to develop efficient and effective deep learning algorithms for computational imaging and detection applications.

Keywords: `Deep learning`;  `Object detection`;  `X-Ray image`.
<br>

References
-----
- Caijing Miao, Lingxi Xie, Fang Wan, Chi Su, Hongye Liu, Jianbin Jiao, Qixiang Ye, "SIXray:A Large-scale Security Inspection X-ray Benchmark for Prohibited Item Discovery in Overlapping Images".

- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, "You Only Look Once: Unified, Real-Time Object Detection".

- Joseph Redmon, Ali Farhadi, "YOLOv3: An Incremental Improvement".

- Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".

Progress
-----
#### YOLOv3:

> The installation and configuration of YOLOv3 have been completed and preliminary test has been carried out. 

> Details of the installation and configuration can be found at https://pjreddie.com/darknet/yolo/ and https://bbs.csdn.net/topics/392556090?list=lz.


 
> |![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/predictions2.jpg)|![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/predictions%201.jpg)|![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/predictions.jpg)|
> |---|---|---|

> Important codes:
> activate detection:
> ```python
> ./darknet detect cfg/yolov3.cfg yolov3.weights
> ```
