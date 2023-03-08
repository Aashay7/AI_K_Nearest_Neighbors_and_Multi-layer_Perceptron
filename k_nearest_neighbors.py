# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Author: Aashay Gondalia


import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """
    # REFERENCE : Implemented the KNN Algorithm from scratch in my Applied Machine Learning Class (CSCI-P 556)
    # Link to Notebook : https://colab.research.google.com/drive/1QMGIPRnZG7UxOwRBjnBFHaXRiYdJmCu3?usp=sharing

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance


    ############################################################################################################################################################
    #   NEW HELPER METHODS DEFINED FOR KNN
    #   *  _apply_weight :          Method to apply weights 'uniform' or 'distance'
    #   *  _nearest_neighbors :     Method to get the nearest_indices and nearest_distances of the points in X_train. 
    ############################################################################################################################################################


    def _apply_weight(self, distances):
        """
        If weight is 'uniform' :: Then Generate neighbors weights which are uniformly distributed.
        
        Else if weight is 'distance' ::  Then Generate neighbors weights which are inversely proportional 
        to the distances from object to its neighbor.

        Args:
            distances(ndarray): 2D numpy array
                first axes is objects
                second axes is neighbors of object
                in the cell (i; j) there is distance from
                the i^th object to its j^th nearest neighbor
        Return:
            weights(ndarray):   array of weights
        """ 

        if self.weights == 'uniform' : 
            weights = np.ones(shape=distances.shape)
        else:
            weights = np.zeros(shape=distances.shape)
            # To counter the division by zero error
            weights = 1. / (distances + 1)  # +1 to avoid Division by Zero Error.
        return weights


    def _nearest_neighbors(self, X):
        """
        Get the nearest_indices and nearest_distances of the points in X_train (self._X) from all point in the test set (X).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.

        Returns:
            nearest_indices : A numpy array of shape (n_samples, n_neighbors) representing the nearest indices of points in the train set.
            nearest_distances : A numpy array of shape (n_samples, n_neighbors) representing the nearest distance of points in the train set.

        """
        nearest_indices = np.zeros(shape=(X.shape[0], 
                                          self.n_neighbors), dtype=np.int) - 1
        nearest_distances = np.zeros_like(nearest_indices)
        
        # Iterate through the Test set examples to get the K nearest points to a given point, 
        # Then populate the nearest_indices matrix, which contains the indices of the K nearest points 
        # from the trainset.
        # Similarly populate the nearest_distances matrix, which contains the distances of the K nearest
        # points from the trainset.
        for i in range(X.shape[0]):
            distances = self._distance(X[i], self._X)
            index_order = np.argsort(distances)[:self.n_neighbors]
            nearest_indices[i] = index_order
            nearest_distances[i] = distances[index_order]
            
        return (nearest_indices, nearest_distances)


    ############################################################################################################################################################
    #   FIT PREDICT METHODS.
    ############################################################################################################################################################


    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        # Training phase is essentially storing the data. 
        # A better approach is to store the train data in a K-D Tree for fast retrievals during testing phase.
        self._X = X
        self._y = y
        return self


    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        # Initialize y by assigning -1 to the labels.
        y = np.zeros(X.shape[0]) - 1       

         # Get the nearest_indices and nearest_distances from the helper method (_nearest_neighbors)
        nearest_indices, nearest_distances = self._nearest_neighbors(X)    

        # nearest_labels are the labels belonging to the nearest points to the test example.
        nearest_labels = self._y[nearest_indices]           
        
        # Apply weight ('uniform' or 'distance', where we transformed the weight vector based on uniform weightage or distance based inverse weightage.)
        weights = self._apply_weight(nearest_distances)    
        
        for i in range(X.shape[0]):     
            # np.bincount -> Counts the occurence of each element in the array. Used in voting.
            counts = np.bincount(nearest_labels[i])

            # Apply the computed weights to the votes from the counts vector. 
            weighted_sum = counts[nearest_labels[i]] * weights[i]

            # Assign the most voted train label to the test label of the given test example.
            y[i] = nearest_labels[i][np.argmax(weighted_sum)]
        
        return y
