# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Author : Aashay Gondalia

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """


    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None
        self._h_layer_ip = None
        self._o_layer_ip = None
        self._h_layer_op = None
        self._o_layer_op = None


    ############################################################################################################################################################
    #   NEW HELPER METHODS DEFINED FOR MLP
    #   *  _initialize    :     Initialization of the hidden and output weights, X, and y.
    #   *  _standardize_X :     The function to standardize the data. 
    #   *  forwardPass    :     The Forward propogation phase in the neural network, where the values are predicted based on the current weights.
    #   *  backwardPass   :     The backward propogation of error through the network happens in this function. 
    ############################################################################################################################################################


    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X = X
        self._y = one_hot_encoding(y)
        #self._X = self._standardize_X(self._X)        # Data Standardization
    
        # WEIGHT INITIALIZATION : Switched from Random to Xavier and He initialization.
        # REFERENCE : https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

        if self.hidden_activation.__name__ in ['sigmoid', 'tanh']:
            # Initialization of weights based on Xavier Initialization.
            l, u = -(1.0 / np.sqrt(self._X.shape[1])), (1.0 / np.sqrt(self._X.shape[1]))
            self._h_weights = l + (np.random.rand(self._X.shape[1],self.n_hidden) * (u - l))
        
        elif self.hidden_activation.__name__ in ['relu']:
            # Initialization of weights based on He Weight initialization. (Gaussian probability)
            std = np.sqrt(2.0 / self._X.shape[1])
            self._h_weights = np.random.rand(self._X.shape[1],self.n_hidden) * std

        else:
            # Initializing random weights to the hidden layer and zeros to the bias vector.
            self._h_weights = np.random.rand(self._X.shape[1],self.n_hidden)

        self._h_bias = np.zeros((1, self.n_hidden))
        
        if self._output_activation.__name__ in ['sigmoid', 'tanh']:
            # Initialization of weights based on Xavier Initialization.
            l, u = -(1.0 / np.sqrt(self.n_hidden)), (1.0 / np.sqrt(self.n_hidden))
            self._o_weights = l + (np.random.rand(self.n_hidden, self._y.shape[1]) * (u - l))
        
        elif self._output_activation.__name__ in ['relu']:
            # Initialization of weights based on He Weight initialization. (Gaussian probability)
            std = np.sqrt(2.0 / self.n_hidden)
            self._o_weights = np.random.rand(self.n_hidden, self._y.shape[1]) * std

        else:
            # Initializing random weights to the output layer and zeros to the bias vector.
            self._o_weights = np.random.rand(self.n_hidden, self._y.shape[1])

        self._o_bias = np.zeros((1, self._y.shape[1]))
        
        
        np.random.seed(42)


    def _standardize_X(self, X):
        """
        The function to standardize the data. 
        Implementation of StandardScaler.
        """

        # The standardized data is (data - mean(data)) / standard_deviation(data)
        return (X - np.mean(X, axis=0)) / np.std(X,axis=0)


    def forwardPass(self, iter):
        """
        The Forward propogation phase in the neural network, where the values are predicted 
        based on the current weights in the hidden and output layers and activation functions.

        """
        
        # Input Layer to Hidden Layer.
        self._h_layer_ip = np.dot(self._X, self._h_weights) + self._h_bias
        self._h_layer_op = self.hidden_activation(self._h_layer_ip)
        
        # Hidden Layer to Output Layer.
        self._o_layer_ip = np.dot(self._h_layer_op, self._o_weights) + self._o_bias
        self._o_layer_op = self._output_activation(self._o_layer_ip)

        # Calculate the error.
        error = self._loss_function(self._y, self._o_layer_op)
        if iter % 20 == 0:
            self._loss_history.append(error)
            print('Loss : ', error)


    def backwardPass(self):
        """
        The backward propogation of error through the network happens in this function. 
        The weights of the hidden and output layers are updated in this phase, such that 
        it leads to lesser error / better prediction.

        """
        
        # Calculating the delta of the output layer which is used in calculating the weights and bias of the output layer. 
        o_layer_delta       =  (self._o_layer_op -  self._y) * self._output_activation(self._o_layer_ip, derivative=True)
        o_weights_update    = np.dot(self._h_layer_op.T, o_layer_delta) 
        o_bias_update       = np.sum(o_layer_delta, axis=0, keepdims=True)
        
        # Update step : Output layer (weights and biases) are updated.
        self._o_weights -=  self.learning_rate * o_weights_update
        self._o_bias    -=  self.learning_rate * o_bias_update

        # Calculating the delta of the hidden layer which is used in calculating the weights and bias of the hidden layer. 
        h_layer_error       = np.dot(o_layer_delta, self._o_weights.T)
        h_layer_delta       = h_layer_error * self.hidden_activation(self._h_layer_ip, derivative=True)
        h_weights_update    = np.dot(self._X.T, h_layer_delta) 
        h_bias_update       = np.sum(h_layer_delta, axis=0, keepdims=True)
        
        self._h_weights -=  self.learning_rate * h_weights_update
        self._h_bias    -=  self.learning_rate * h_bias_update
    

    ############################################################################################################################################################
    #   FIT PREDICT METHODS.
    ############################################################################################################################################################


    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        
        self._initialize(X, y)
        
        for iter in range(self.n_iterations):
            self.forwardPass(iter)
            self.backwardPass()

    
    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        
        #X = self._standardize_X(X)         # Standardize Test data (leakage issue)

        # Forward Propagation of the test set through the network
        h_pred = np.dot(X, self._h_weights) + self._h_bias
        h_pred = self.hidden_activation(h_pred)

        # Hidden Layer to Output Layer.
        o_pred = np.dot(h_pred, self._o_weights) + self._o_bias
        o_pred = self._output_activation(o_pred)

        # Prediction of the test set.
        y_pred = np.argmax(o_pred, axis=1)
        return y_pred