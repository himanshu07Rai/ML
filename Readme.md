## Machine Learning Model

A Machine learning model is a function that tries to find the relationship between the **Features **and the **Target variable**.

It tries to find pattern in the data, understand the data and trains on the data. Based on this learning, a Machine Learning Model makes **Predictions **an recognize patterns

### Supervised Learning : 

Model learns from labelled data

![image](https://user-images.githubusercontent.com/64080063/127961106-de1d826d-ac96-4470-b76a-1ccd45aec30f.png)



### Unsupervised Learning:

Model learns from unlabelled data

![image](https://user-images.githubusercontent.com/64080063/127961403-5bb7087a-501f-4a16-b81b-534a1c06b2ad.png)



## Overfitting

Overfitting refers to a model that models the training data too well. Overfitting happens when a model learns the details and noise in the training dataset to the extent that it negatively impacts the performance of the model.

**Sign **: High raining data accuracy and very low test data accuracy

**Cause** :

* Less data
* Increased Complexity of the model
* More number of layers in Neural Network

**Preventing Overfitting** :

* Using more data
* Reduce number of layers in Neural Network
* Use Dropout Layers
* Early Stopping
* Bias - Variance Tradeoff



## Underfitting

Underfitting happens when the model does not learn enough from the data. Underfitting occurs when a machine learning model cannot capture the underlying trend of the data.

**Sign** : Very low training data accuracy

**Causes** :

* Choosing a wrong model
* Less complexity of the model
* Less variance but high bias

**Preventing Underfitting** : 

* Choosing correct model
* Increasing the complexity of the model
* Mode number of parameters to the model
* Bias - Variance Tradeoff



## Bias - Variance Tradeoff

**Bias** is the difference between the average prediction of the model and the correct value which we are trying to predict.

![image](https://user-images.githubusercontent.com/64080063/128454991-c06581ee-dbff-4653-bc13-95427f19f12b.png)



**Variance** is the amount that the estimate of the target function will change if different training data was used.



**Underfitting** :  High Bias & Low Variance

**Overfitting** : Low Bias & High Variance

**Bias-Variance Tradeoff** : 

![image](https://user-images.githubusercontent.com/64080063/128455459-2f2b39f7-f60f-4ef6-8998-f7b0560bd850.png)

​					

**Techniques to have better Bias-Variance Tradeoff**

* Good Model Selection
* Regularization ( Reducing value of coeffiients)
* Dimensionality Reduction
* Ensemble method ( multiple models together )



## Loss Function

Loss Function measures how far an estimated value is from its true value. It is helpful to determine which model performs better & which parameters are better. 

![image](https://user-images.githubusercontent.com/64080063/128456030-3c02feb4-7344-4179-9a8a-334c6bba1cb9.png)

**Types** : 

* Cross Entropy
* Squared Error
* KL Divergence



Low Loss value == High Accuracy



## Evaluation Matrics

How well our model performs

* **Accuracy Score** : ratio of number of correct predictions to the total number of input data points 
* **Mean Squared Error** : the average squared difference between the estimated values and the actual value









## Artificial Neural Networks

Because the signal just goes from lest to right, we call it a **feedforward** neural network.

### Forward Propagation

- Logistic Regression : Computer simulation of a neuron
- Every neuron works in similar fashion
- Two important ways to extend the single neuron:
  - The same input can be fed to multiple different neurons , each calculating something different ( more neurons per layer / Makes layer deep ).
  - Neurons in one layer can act as inputs to another layer ( Deep network ).

### Automatic Feature Engineering

- W's and b's are randomly initialized , found iteratively using gradient descent.

### ReLU Layers

- ReLU activation function

### Binary Classification

- sigmoid in final layer

### Multi Class Classification

* Softmax function is used
  * tf.keras.layers.Dense(K,actication='softmax')
  * not meant for hidden layer activation

```markdown
|   Task	                    |   Activation Function	|
|   Regression	                |   None / Identity	    |
|   Binary classification	    |       Sigmoid 	    |
|   Muticlass classification	|       Softmax	  	    |
```

<img src="C:\Users\ACER\AppData\Roaming\Typora\typora-user-images\image-20210712112244450.png" alt="image-20210712112244450" style="zoom:50%;" />



**Deep Learning** 
Deep learning refers to the training of things called neural networks. These are inspired by the neurons of the human brain. Building block of a neural network is also a neuron.
Mimicking human brain. 

**Neural network** 
Take in input .Train themselves to understand patterns in the data.Output predictions

**Input layer -------------Several Hidden layers ---------------Output layer**

A neural network receives input , converts the input signal by changing the state using an activation function to produce an output.
Activation function introduces non-linearity in the network.

A neural network will have : 
Input layer with the bias unit, which is 1, It is also referred as the interceptOne or more hidden layers .Each with a bias unit.Output layer .Weights associated with each connection.Activation function which converts an input signal of a note to an output signal.

**Weights** 

Weights are how, neural networks learn. We adjust the weights to determine the strength of the signal.We randomly initialize the weights (w) and multiply them with the input (x).
This is forward propagation.
We add the bias of the layer to it and then apply the activation function on it.
Let y be the total sum + bias, then output = act(y)

**Activation function**

This helps decide if we need to fire ( activate ) a neutron or not and if we need to activate a neutron , what will be the strength of the signal.
It squishes the whole big range of numbers into 0-1 range. (sigmoid )
Activation function is the mechanism by which the neuron process and passes information through the neural network.
Its purpose is to introduce non-linearity in the network
Now, for a neuron , y can range from -infyto +infy. Here, we will not be able to decide if we need to activate (fire) the neuron or not. This is where activation function helps us.

**Bias**
Bias tells us how high the weighted sum needs to be before the neuron starts getting meaningfully activated. It is like a threshold.

**Learning** 
Finding the right weights and biases. (back propagation)













**Convolutional Neural Networks** 
Convolution denotes the mathematical function of convolution, which is a special kind of linear operation where two functions are multiplied to produce a third function which expresses how the shape of one function I modified by the other.
Two images in form of matrices are multiplied to give an output that is used to extract features from the image.

**input : n \* n**

**filter** : f \* f

**padding** : p

**stride** : s
**output : ( n + 2 \* p - f ) / s + 1**
**Dilated Convolution :** Convolution applied to input with defined gaps. We don’t look for contiguous group of pixels
**Multi Channel Convolution** : In color images, we have three channels ( R,G,B ) i.e., every pixel is composed of three components .So , we have 3 different filters for three channels. eg : ![img](https://lh4.googleusercontent.com/Os4xQ86NM1WxObZRpKGNkc_U5ae7EIJgeKbIbvkm22peZBRaRGSxdvOyjBGKDsTF_Vcb0H9FDJOrq8qcsuZbS8Pw0Cx5CXYC7Su6LQgKCxyV0YNiQglqZO1yu974yQ)

Apply particular filter for a particular channel , then do total summation
Input : 3 channelFilter : 3 channelOutput / Result : 1 channel


**Multi-Filter Convolution** **:** 
![img](https://lh6.googleusercontent.com/52IpQYba-VVZh_zZ2UWneh9mTZSyzN-YDnERjFmU0hzV9LlbHzMgdOuBASyR9SsbjQ44izk4CAS7GcTg24k9UXMLQST78OQtGSyiPsXGUO-AtomtJM14MHeVhIJI6Q)
No. of outputs = No. of filters



**CNN Architecture**
There are two main parts to a CNN architecture:
A convolution tool that separates and identifies the various features of the image for analysis in a process called as Feature Extraction.A fully connected layer that utilizes the output from the convolution process and predicts the class of the image based on the features extracted in previous stages.

**Features**
**Convolutional Layer**
First layer.Used to extract the various features from the input images.Here, convolution is performed between the input image and a filter of a particular size MxM.By sliding the filter over the input image, the dot product is taken between the filter and the parts of the input image with respect to the size of the filter (MxM).The output is termed as the Feature map which gives us information about the image such as the corners and edges.This feature map is fed to other layers to learn several other features of the input image.Convolution layers summarizes the presence of features in an input image.

![img](https://lh6.googleusercontent.com/H9FjavpmHjNxiih827s5K7-QtH09AJikj5gxjo0h1tdyWoa7hlUVq3bGQLn7ZtLzbxQStYYX6yvxjAfLXSUAcbkmNspBDUZRt8HUbc3qXOufKqadJwjk015-bASDfg)


**Pooling layer**
Convolutional Layer is followed by a Pooling Layer. The primary aim of this layer is to decrease the size of the convolved feature map to reduce the computational costs. This is performed by decreasing the connections between layers and independently operates on each feature map. Because of this, the model is tolerant towards variations and distortions.We only capture the feature and filter all the noise.Pooling operation is specified, rather than learned.Two common functions used in pooling operations areAverage pooling : Calculate the average value for each patch in the feature map..Max pooling: Calculate the maximum value for each patch of the feature map.The result of using a pooling layer and creating down sampled or pooled feature map is a summarized version the features detected in the input. Depending upon method used, there are several types of Pooling operations.The Pooling Layer usually serves as a bridge between the Convolutional Layer and the FC Layer.Has hyperparameters but no learnable parameter.
**Global Pooling Layers**: Instead of down sampling patches of the input feature map, global pooling down samples the entire feature map to a single map . This would be same as setting the pool size to the size of input feature map.
In the above two layers, feature extraction takes place.

 **Fully Connected Layer**
The Fully Connected (FC) layer consists of the weights and biases along with the neurons and is used to connect the neurons between two different layers.These layers are usually placed before the output layer and form the last few layers of a CNN Architecture.The input image from the previous layers are flattened and fed to the FC layer. The flattened vector then undergoes few more FC layers where the mathematical functions operations usually take place. In this stage, the classification process begins to take place.

**Dropout**
Usually, when all the features are connected to the FC layer, it can cause overfitting in the training dataset. Overfitting occurs when a particular model works so well on the training data causing a negative impact in the model’s performance when used on a new data.To overcome this problem, a dropout layer is utilised wherein a few neurons are dropped from the neural network during training process resulting in reduced size of the model. On passing a dropout of 0.3, 30% of the nodes are dropped out randomly from the neural network.

**Activation Functions**
They are used to learn and approximate any kind of continuous and complex relationship between variables of the network. It decides which information of the model should fire in the forward direction and which ones should not at the end of the network.It adds non-linearity to the network.There are several commonly used activation functionsReLU, Softmax, tanH, Sigmoid functions. Each of these functions have a specific usage. 
	![img](https://lh6.googleusercontent.com/BE0UW5QBi2Wyh_0kUYLYaD9I5VcXPvf7K0ujPgqxJu2qvWhgSeMMBefnNv5uV7PQCvQLmhGJ2mMbxmd5jMBPyELyVSABtBkKffFf4F6dA-WOz7ECFaPUdcwDAPMGXQ)

**Padding** : To make the size of output image same as input image in convolution layer, we use padding.

**Hyperparameters of a classic CNN :** 
Input image size Number of convolutional units size of filters in each layerpaddingstridesnumber of filters in each layersize of pooling filters stride of pooling filtersNumber of fully connected layersnumber of nodes in each FC layer
The core of these developments revolves around the critical question, how to perceive the driving environment with higher certainties.The key technology on which success of AVs depends is how accurately they are able to perceive the driving environment.The initial step in this quest is to recognize the static and dynamic objects around the vehicles with higher accuracies.Static objects (road, speed breakers, traffic light, buildings) and dynamic objects (cars,cycles, trucks) Driving surface identification is an important and critical task in the overall success of AVs

**Data Augmentation**

MirroringZoom +/-

Translation

Rotation

Shearing

Distortion

Brightness

Color Shifting

 
