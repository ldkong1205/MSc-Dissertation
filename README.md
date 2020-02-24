# 🎓 MSc Dissertation

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



# Chapter 1  Introduction

### 1.1  Background
.

### 1.2  Motivation
.

# Chapter 2  Deep Learning

### 2.1  Overview
.


### 2.2  Logistic Regression
.


### 2.3  Activation Functions

The performance of a deep learning model can be very different due to the utilization of different activation functions. In this section, we will discussion some commonly used activations functions and their derivatives.

### 2.3.1 Sigmoid Activation Function

From the previous discussion of the logistic regression, we know that for a binary classification problem, the sigmoid activation function is the first choice.

g(z)=1/(1+e^(-z) )

d/dz g(z)=1/(1+e^(-z) ) (1-1/(1+e^(-z) ))=g(z)(1-g(z))

### 2.3.2 Tanh Activation Function

g(z)=(e^z-e^(-z))/(e^z+e^(-z) )

d/dz g(z)=1-((e^z-e^(-z))/(e^z+e^(-z) ))^2=1-(g(z))^2

### 2.3.3 ReLU Activation Function

g(z)=max⁡(0,z)


### 2.4  Gradient Descent for Neural Networks

.

### 2.4.1 Neural Network Architecture
.

### 2.4.2 Forward Propagation
.

### 2.4.3 Backward Propagation
.


### 2.5 Optimization
.

### 2.5.1 Bias and Variance
.

### 2.5.2 Regularization

For a model with high variance, the most useful and reliable way to improve its performance is adding more training examples. But getting more data might be quite difficult sometimes. When facing this, we will need a technique to alleviate the overfitting problem without feeding extra data, and regularization turns out to be helpful. 
Recall that for logistic regression, our goal is to reduce the overall cost of our model, which can be formulated as follows:

min┬(w,b)⁡〖J(w,b)〗=min┬(w,b)⁡〖1/m ∑_(i=1)^m▒L(y ̂^((i) ),y^((i) ) ) 〗

To implement regularization, we add an extra term into the objective function above, i.e.,

min┬(w,b)⁡J(w,b)=min┬(w,b)⁡(1/m ∑_(i=1)^m▒L(y ̂^((i) ),y^((i) ) ) +Γ)

+ L1 regularization:

Γ=λ/m |w|_1=λ/m ∑_(j=1)^(n_x)▒|w_i | 

+ L2 regularization:

Γ=λ/2m ‖w‖_2^2=λ/2m ∑_(j=1)^(n_x)▒w_j^2 =λ/2m w^T w

where | · |_1 and ‖ · ‖_2 denote the one-norm and two-norm of a vector, respectively, and λ is an adjustable parameter.

The reason that we only implement the regularization on weight parameter w but not on bias parameter b is that comparing with the high dimensions of w, the dimension of b is relatively small, maybe just a single number. So almost all the parameters are in w rather b and adding another regularization term like λ/2m b^2 just won’t make much difference in practice.

According to some research results, when using L1 regularization, the weight parameter w will end up being sparse, that is to say, w will have more zeros in it. In practice, this might help with compressing the network, but only contribute a little to the optimization. Therefore, L2 regularization is used much more often and achieves better results.

For the regularization of a L-layer neural network, the cost function is expanded to l dimensions, which much more complicated than the logistic regression. The L2 regularization of such a network can be described as the following equation:

J(w^[1] ,b^[1] ,⋯,w^[l] ,b^[l]  )=1/m ∑_(i=1)^m▒L(y ̂^((i) ),y^((i) ) ) +λ/2m ∑_(l=1)^L▒‖w^[l]  ‖_F^2 

‖w^[l]  ‖_F^2=∑_(i=1)^(n^[l-1] )▒∑_(l=1)^(n^[l] )▒〖(w_ij^([l]))〗^2 



# Appendices

### A.  Implementations of Logistic Regression
### A.1  Packages
In order to implement the logistic regression using Python 3, the following packages are needed:
+ `numpy`: is the fundamental package for scientific computing with Python.
+ `h5py`: is a common package to interact with a dataset stored on an H5 file.
+ `matplotlib`: is a famous library to plot graphs in Python.
+ `PIL` and `scipy`: are used to test the models with images.

      import numpy as np
      import matplotlib.pyplot as plt
      import h5py
      import scipy
      from PIL import Image
      from scipy import ndimage
      from lr_utils import load_dataset
      
      %matplotlib inline

### A.2 	Overview of the Problem Set

Our example dataset containing:
+	A training set of `m_train` images labeled as cat (y=1) or non-cat (y=0).
+ A test set of `m_test` images labeled as cat (y=1) or non-cat (y=0).
+ Each image is of shape `(num_px, num_px, 3)`, where 3 is for the three channels (RGB). Thus, each image is square `(height = num_px)` and `(width = num_px)`.

Our goal is to build an image-recognition algorithm using logistic regression that can correctly classify images as cat of non-cat.

First, load the dataset by the following code:

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()`

Here we use added “`_orig`” at the end of the image datasets because we are going to preprocess them. Each line of our `train_set_x_orig` and `test_set_x_orig` is an array representing an image. We can visualize the 200th example by the following code:

    index = 200
    plt.imshow(train_set_x_orig[index])
    print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture.")
  
The following code is used to extract the size of the images inputted:

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

Through the outputs we known that the dataset contains 209 training examples, 50 test examples, and each input image is of the size 64×64×3.

For convenience, we should now reshape images of shape `(num_px, num_px, 3)` into a numpy array of shape `(num_px∗num_px∗3, 1)`. After this, our training and test datasets become numpy arrays where each column represents a flattened image. There should be `m_train` (respectively `m_test`) columns. To achieve this, the following code is needed:

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

The shapes of our `train_set_x` and `test_set_x` become (12288, 209) and (12288, 50), respectively.

To represent color images, the red, green and blue channels must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255. One common preprocessing step in machine learning is to center and standardize the dataset, which can be achieved by:

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

### A.3 	General Architecture of the Learning Algorithm

In this subsection, we will build a logistic regression using neural network mindset. First, the following mathematical expressions are used to describe the algorithm:
For one input example x^((i)):

z^((i))=w^T x^((i))+b
y ̂^((i))=σ(z^((i) ) )=sigmoid(z^((i)))
L(y ̂^((i) ),y^((i) ) )=-y^((i) )  log⁡(y ̂^((i) ) )-(1-y^((i) ))log⁡(1-y ̂^((i) ))
The cost is then computed by summing over all the training examples:
J=1/m ∑_(i=1)^mL(y ̂^((i) ),y^((i) ) ) 



