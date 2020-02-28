|![image](https://github.com/ldkong1205/MSc-Dissertation/blob/master/IMAGE/ntu_logo.png)|
|---|


# Chapter 2 Deep Learning

# 📚 Content

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
