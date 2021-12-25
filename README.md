# Machine Learning.

## Part 1 : K-Nearest Neighbours Classification.


### 1.1 Problem Statement

In the machine learning world, k-nearest neighbors is a type of non-parametric supervised machine learning
algorithm that is used for both classification and regression tasks. 

For classification, the principle behind k-nearest neighbors is to find k training samples that are closest in distance to a new sample in the test
dataset, and then make a prediction based on those samples.

These k closest neighbors are used to try and predict the correct discrete class for a given test sample.
This prediction is typically done by a simple majority vote of the k nearest neighbors of each test sample; in other words, the test sample is assigned the data class which has the most representatives within the k nearest neighbors of the sample. 

An alternative method for prediction is to weigh the neighbors such that the nearer neighbors contribute more to the fit than do the neighbors that are further away. For this, a common choice is to assign weights proportional to the inverse of the distance from the test sample to the neighbor.


### 1.2 Data

The starter datasets provided for the assignment are
- IRIS dataset    -  3 output classes
- DIGIT dataset   - 10 output classes

### 1.3 Approach and Methodology

FIT : 
- As per the KNN algorithm, the fit method essentially captures / stores the train data. 

PREDICT :
- The predict method calls the nearest_neighbors method to get the nearest_indices and the nearest_distances for a given test set with relation to the stored train set. 
- The apply_weight method applies 'uniform' or 'distance' (inverse distance) weights to get the indexed weights vector.
- For each entry in the test set, the comparison is made based on the distance and the k nearest elements are found.
- The labels of these k nearest neighbors are checked and the weight is applied according to them. (Nearer neighbor has more weightage if 'distance' is selected, otherwise all votes have an equal say in the assignment. ). The most occuring label after applying weightage (in case of 'distance') or no weightage (in case of 'uniform') is selected and assigned to the test example. 


### 1.4 Results

- The KNN model is outstanding at predicting the variable with high accuracy. In multiple cases, the KNN model surpasses the performance (in terms of accuracy) of the SKlearn model. 
- The whole test results are present in the html file in the repository.



## Part 2 : Multilayer Perceptron Classification.

### 2.1 Problem Statement

#### 2.1.1 Description

In machine learning, the field of artificial neural networks is often just called neural networks or multilayer
perceptrons. As we have learned in class, a perceptron is a single neuron model that was a precursor to the
larger neural networks that are utilized today.

The building blocks for neural networks are neurons, which are simple computational units that have input
signals and produce an output signal using an activation function. Each input of the neuron is weighted
with specific values, and while the weights are initially randomized, it is usually the goal of training to
find the best set of weights that minimize the output error. 

The weights can be initialized randomly to small values, but more complex initialization schemes can be used that can have significant impacts on the
classification accuracy of the models. 

A neuron also has a bias input that always has a value of 1.0 and it
too must be weighted. These weighted inputs are summed and passed through an activation function, which
is a simple mapping that generates an output value from the weighted inputs. Some common activation
functions include the sigmoid (logistic) function, the hyperbolic tangent function, or the rectified linear unit
function.

![image](https://media.github.iu.edu/user/18070/files/5b195600-58fd-11ec-93d9-77ba852f11b6)


These individual neurons are then arranged into multiple layers that connect to each other to create a
network called a neural network (or multilayer perceptron). The first layer is always the input layer that
represents the input of a sample from the dataset. The input layer has the same number of nodes as the
number of features that each sample in the dataset has. 

The layers after the input layer are called hidden layers because they are not directly exposed to the dataset inputs. The number of neurons in a hidden layer
can be chosen based on what is necessary for the problem. The neurons in a specific hidden layer all use
the same activation function, but different layers can use different ones. Multilayer perceptrons must have
at least one hidden layer in their network.

The final layer is called the output layer and it is responsible for outputting values in a specific format

It is common for output layers to output a probability indicating the chance that a sample has a specific target
class label, and this probability can then be used to make a final clean prediction for a sample. For example, if
we are classifying images between dogs and cats, then the output layer will output a probability that indicates
whether dog or cat is more likely for a specific image that was inputted to the neural network. 

The nature of the output layer means that its activation function is strongly constrained. 
- Binary classification problems have one neuron in the output layer that uses a sigmoid activation function to represent the probability
of predicting a specific class. 
- Multi-class classification problems have multiple neurons in the output layer,
specifically one for each class. 

In this case, the softmax activation function is used to output probabilities for each possible class, and then you can select the class with the highest probability during prediction.

#### 2.1.2. One Hot Encoding
Before training a neural network, the data must be prepared properly. Frequently, the target class values are
categorical in nature: for example, if we are classifying pets in an image, then the possible target class values
might be either dog, cat, or goldfish. 

![image](https://media.github.iu.edu/user/18070/files/4341d200-58fd-11ec-8a1e-7ef02b5f99f6)

However, neural networks usually require that the data is numerical.
Categorical data can be converted to a numerical representation using one-hot encoding. One-hot encoding
creates an array where each column represents a possible categorical value from the original data (for the
image pet classification, one-hot encoding would create three columns).Each row then has either 0s or 1s in
specific positions depending on the class value for that row. Here is an example of one-hot encoding using
the dog, cat, or goldfish image classification example, where we are using five test samples and looking at
their target class values.

### 2.2 Data

The starter datasets provided for the assignment are
- IRIS dataset    -  3 output classes
- DIGIT dataset   - 10 output classes

### 2.3 Approach and Methodology

#### 2.3.1 Method Definition

For the Multilayer perceptron, multiple helper methods have been defined and their functions are listed as below.
- _initialize    :     Initialization of the hidden and output weights, X, and y.
- _standardize_X :     The function to standardize the data. 
- forwardPass    :     The Forward propogation phase in the neural network, where the values are predicted based on the current weights.
- backwardPass   :     The backward propogation of error through the network happens in this function. 
 

#### 2.3.2 Initialization of Weights

It was noticeable that choosing the correct starting weights affected the accuracy positively. 
- In the case of Linear/Identity activation function, the associated layer weights are randomly initalized.

- In the case of Sigmoid/Logistic activation and the Tanh/Hyperbolic activation functions, the associated layer weights are initialized based on Xavier initialization. 
  - lower, upper = -(1/sqrt(n)), 1/sqrt(n)   ;  where n is the number of neurons in the previous layer.
  - layer_weights = lower + random(layer weight matrix) * (upper - lower)
 
 ![image](https://media.github.iu.edu/user/18070/files/fbbc4580-58fe-11ec-9edc-1e10dfffffa0)

- In the case of ReLU activation, the associated layer weights are initialized based on He Initialization 
  - std = sqrt(2. / n)  ; where n is the number of neurons in the previous layer.
  - layer_weights = random(layer weight matrix) * std
 
 ![image](https://media.github.iu.edu/user/18070/files/535ab100-58ff-11ec-8dc2-1e5874d8449e)

Reference : https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

#### 2.3.3 One-Hot Encoding

In the one-hot encoding method, following line is executed. 

-   return np.array([(y == cat) * 1. for cat in np.unique(y)]).T

it can be broken down as below.

- The np.unique(y) == Unique categories in the feature.
- cats = np.unique(y)
- mtx = np.zeros((y.shape[0], len(cats)))
- (y == cat) * 1 gives us the array with value == 1 where the element == category
- Run the above logic for all unique elements present in y. 
- Convert to numpy.ndarray and then transpose the matrix to get the one-hot encodings in the desired format.

#### 2.3.4 Forward Pass / Forward Propogation

In the forward pass, the data is passed forward into the layer till the output layer. 
The following operations happen at each layer. 
- A.  Layer_input  = X * W + b    ;  X is the data, W is the layer weight, b is the layer bias
- B.  Layer_output = layer_activation_function(Layer_input)     ; layer_activation_function in the given problem (for hidden layers : ['identity', 'sigmoid', 'tanh', 'relu'] and for output layer : 'softmax' )

At the output layer, the error is calculated. 

##### 2.3.4.1 Cross Entropy Loss

Cross Entropy loss is used in the classification problems to accurately drive the learning effectively. 

![image](https://media.github.iu.edu/user/18070/files/3b842c80-5901-11ec-9feb-c3bcc8cddd0b)


#### 2.3.5 Backward Pass / Backpropogation

In the backward pass (Backpropogation), the error is propogated backwards in to the network and the weights and biases of each layer are updated. 
The Learning rate is a hyperparameter that maintains the intensity at which the weight update takes place. 

Higher learning rate often leads to divergence after converging upto a certain point. It may never reach a local optimum. 
Selecting the best learning rate is a key part of the neural network learning. 

Here, we are experimenting with 3 learning rates (0.1, 0.01, 0.001). It is clear from the results that the learning rate selected as 0.1 leads to a poorly learnt model. 


#### 2.3.6 Activation Functions

- Linear / Identity Activation Function 

![image](https://media.github.iu.edu/user/18070/files/d5000e00-5902-11ec-9dc6-8caf8211fff5)



- Sigmoid / Logistic Activation Function

![image](https://media.github.iu.edu/user/18070/files/80f52980-5902-11ec-9303-815084a7820b)
<img src="https://media.github.iu.edu/user/18070/files/e9440b00-5902-11ec-82f3-87e86aa69990" alt="drawing" width="300"/>



- Tan-h / Hyperbolic Activation Function

![image](https://media.github.iu.edu/user/18070/files/936f6300-5902-11ec-952c-ceb177333dbd)
<img src="https://media.github.iu.edu/user/18070/files/f2cd7300-5902-11ec-90cf-ba5d3235fec0" alt="drawing" width="300"/>



- ReLU (Rectified Linear Unit) Activation Function

![image](https://media.github.iu.edu/user/18070/files/a6823300-5902-11ec-9649-2f9b270a6e06)
<img src="https://media.github.iu.edu/user/18070/files/ffea6200-5902-11ec-9b16-b1b69a396687" alt="drawing" width="300"/>




- Softmax Activation Function

![image](https://media.github.iu.edu/user/18070/files/b13cc800-5902-11ec-82e1-239d73ca63c0)
<img src="https://media.github.iu.edu/user/18070/files/08db3380-5903-11ec-97ad-3b69d1b7e996" alt="drawing" width="300"/>



### 2.4 Challenges

I faced numerous challenges in the backpropogation method. 
After refering to the articles and the NNFS book, I was able to implement the backpropogation logic. 
Apart from the occassional dimensionality mismatch errors, it was a straightforward implementation.


### 2.5 Results

It was noticeable that the 0.1 learning rate fails to converge effectively compared to 0.01 and 0.001. 
The accuracy achieved after being boosted with the help of He and Xavier weight initialization was close to the ones achieved by the SKLearn MLP model. 
The whole test results are present in the html file in the repository.

### 2.6 References

https://medium.com/@snaily16/what-why-and-which-activation-functions-b2bf748c0441
https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
https://colab.research.google.com/drive/1QMGIPRnZG7UxOwRBjnBFHaXRiYdJmCu3?usp=sharing
https://vitalflux.com/cross-entropy-loss-explained-with-python-examples/
https://nnfs.io/



