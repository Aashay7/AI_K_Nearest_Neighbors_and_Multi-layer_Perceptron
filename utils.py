# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Author : Aashay Gondalia



import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    # Euclidean_distance(x1, x2) = (x1 - x2)**2, which can be implemented using the norm 
    # method in the np.linalg package. 
    # The 'ord' parameter specifies the order of the norm, which in this case is 2.
    return np.linalg.norm(x1 - x2, ord=2, axis=0)
    

def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    # Manhattan_distance(x1, x2) = |x1 - x2|, which can be implemented using the norm 
    # method in the np.linalg package.
    # # The 'ord' parameter specifies the order of the norm, which in this case is 1.
    return np.linalg.norm(x1 - x2, ord=1, axis=0)
    


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    # Reference : https://medium.com/@snaily16/what-why-and-which-activation-functions-b2bf748c0441
    # From the article -> Linear or Identity Activation function.
    # Equation:     f(x) = x
    # Derivative:   f’(x) = 1
    # Range:        (-∞, +∞)

    if derivative:
        # If derivative is True, Then return the derivative of the activation function which is f'(x) = 1. 
        # Hence the matrix of computed derivative of the activation function i.e. a matrix of same shape as x, 
        # with all values equal to 1
        return 1
    
    # If derivative is False, Then return the identity activation function of the given input data x which is f(x) = x
    # (the x matrix itself).
    return x


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    # Reference : https://medium.com/@snaily16/what-why-and-which-activation-functions-b2bf748c0441
    # From the article -> Sigmoid or Logistic Activation function.
    # Equation:     f(x) = s= 1/(1+e⁻ˣ)
    # Derivative:   f’(x) = s*(1-s)
    # Range:        (0,1)

    sigmoid = 1. / (1. + np.exp(-x))
    if derivative:
        # If derivative is True, Then return the derivative of the sigmoid activation function 
        # which is f'(x) = s * (1-s), where s =  1 / (1 + e^(-x))
        return sigmoid * (1. - sigmoid)

    # If derivative is False, return the sigmoid activation function of the given input data x, 
    # given by : 1 / (1 + e^(-x))
    return sigmoid


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    # Reference : https://medium.com/@snaily16/what-why-and-which-activation-functions-b2bf748c0441
    # From the article -> Tan-h or Hyperbolic Activation function.
    # Equation :    f(x) = a = tanh(x) = 2 / (1 + e^-2x) - 1
    # Derivative:   f'(x) = (1- a²)
    # Range:        (-1, 1)

    tanh = 2. / (1. + np.exp(-2 * x)) - 1.
    if derivative:
        # If derivative is True, then return the derivative of the Hyperbolic activation function
        # which is f'(x) = (1 - a^2) where a = 2 / (1 + e^-2x) - 1
        return (1. - (tanh ** 2))

    # If derivative is False, return the tanh / hyperbolic activation function of the given input data x,
    # given by : 2 / (1 + e^-2x) - 1
    return tanh



def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    # Reference : https://medium.com/@snaily16/what-why-and-which-activation-functions-b2bf748c0441
    #             https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
    #             
    # From the article ->  ReLU (Rectified Linear Unit) Activation function.
    # Equation: f(x) = a = max(0,x)
    # Derivative: f’(x) = { 1 ; if z>0, 0; if z<0 and undefined if z=0 }
    # Range: (0, +∞)

    if derivative:
        # If the derivative is True, then return the derivative of the ReLU activation function, 
        # which is f'(x) = { 1 ; if z>0, 0; if z<0 and undefined if z=0 }
        return np.where(x >= 0, 1, 0)
        #return 1 * (x > 0)

    # If the derivative is False, then return the ReLU activation function of the gien input data x,
    # given by : max(0,x)   ::: which can be achieved by  x * (x > 0)
    return np.where(x >= 0, x, 0)
    #return x * (x > 0)


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))



def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    # Reference: Applied Machine Learning Course Notebooks. 
    # Cross Entropy Loss 
    m = y.shape[0]
    p = np.clip(p, 1e-15, 1 - 1e-15)

    if len(np.unique(y, axis=1)) > 2:
        # Categorical Cross Entropy
        return (-1./m) * np.sum(y * np.log(p))
    else:
        # Binary Cross Entropy
        return (-1./m) * np.sum((y * np.log(p)) + ((1-y) * np.log(1-p)))


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    # np.unique(y) == Unique categories in the feature.
    # cats = np.unique(y)
    # mtx = np.zeros((y.shape[0], len(cats)))
    # (y == cat) * 1 gives us the array with value == 1 where the element == category
    # Run the above logic for all unique elements present in y. 
    # Convert to numpy.ndarray and then transpose the matrix to get the one-hot encodings in the desired format.
    return np.array([(y == cat) * 1. for cat in np.unique(y)]).T
