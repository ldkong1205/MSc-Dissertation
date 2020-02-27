|![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/ntu_logo.png)|
|---|

🏡 Back to [Homepage](https://github.com/ldkong1205/MSc-Dissertation/blob/master/README.md)

# CONTENT

### A. Image Recognition using Logistic Regression

&emsp; A.1 - Packages

&emsp; A.2 - Overview of the Problem Set

&emsp; A.3 - General Architecture of the Learning Algorithm

&emsp; A.4 - Building the Parts

&emsp;&emsp;&emsp; A.4.1 - Helper Function

&emsp;&emsp;&emsp; A.4.2 - Initializing Parameters

&emsp;&emsp;&emsp; A.4.3 - Forward and Backward Propagations

&emsp;&emsp;&emsp; A.4.4 - Optimization

&emsp;&emsp;&emsp; A.4.5 - Prediction

&emsp; A.5 - Merge all Functions into a Model



### B. Planar Data Classification with a Hidden Layer

&emsp; B.1 - Packages

&emsp; B.2 - Dataset

&emsp; B.3 - Simple Logistic Regression

&emsp; B.4 - Neural Network

&emsp;&emsp;&emsp; B.4.1 - Defining the Structure

&emsp;&emsp;&emsp; B.4.2 - Initializing Parameters

&emsp;&emsp;&emsp; B.4.3 - The Loop

&emsp;&emsp;&emsp; B.4.4 - Integration

&emsp;&emsp;&emsp; B.4.5 - Prediction



### C. Building a Deep Neural Network: Step by Step

&emsp; C.1 - Packages

&emsp; C.2 - Initialization

&emsp;&emsp;&emsp; C.2.1 - Two-Layer Neural Network

&emsp;&emsp;&emsp; C.2.2 - L-Layer Neural Network

&emsp; C.3 - Forward Propagation Module

&emsp; C.4 - Cost Function

&emsp; C.5 - Backward Propagation Module



### D. Initialization, Regularization, and Gradient Checking

&emsp; D.1 Initialization: Neural Network Model

&emsp;&emsp;&emsp; D.1.1 Zero Initialization

&emsp;&emsp;&emsp; D.1.2 Random Initialization

&emsp;&emsp;&emsp; D.1.3 He Initialization



### E. Optimization

### F. Building Deep Neural Networks with TensorFlow

### G. Convolutional Module: Step by Step

### H. Residual Networks

### I. Car Detection with YOLO

### J. Face Recognition

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
