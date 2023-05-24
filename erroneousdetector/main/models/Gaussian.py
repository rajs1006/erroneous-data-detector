import bz2
import math
import pickle
import warnings
from copy import deepcopy

import numpy as np
from numba import jit

from erroneousdetector.main.utils.Constants import Resource as resCons
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)

warnings.filterwarnings("ignore")


class Gaussian:
    """ 
    A class to encapsulate the methods for anomaly detection.\n
    Available class methods:
        fit: get the mean and variance of an array (data set);
        detect: obtain the p-value for an observed data point;
        update: update the Gaussian fit;
        save: compress, pickle, and save a file;
        load: load, decompress, and unpickle a file;
        score: calculate precision and recall scores;
        copy: create a deep copy of an object;
    """

    def __init__(self, epsilon):
        """
        Load epsilon from env file to a variable.\n
        Args:
            epsilon (float): Min pValue
        """
        self.eps = epsilon
        self.std = 3

    @jit(parallel=True)
    def fit(self, X):
        """
        Obtain the mean and variance of a given array (data set).\n
        Args:
            X (ndarray):        input array;
        Returns:
            mu (ndarray):       mean value of the given array;
            sigma2 (ndarray):   variance of the given array;
        """
        self.m, n = X.shape
        self.mu = np.mean(X, axis=0).reshape((1, n))
        self.sigma2 = np.var(X, axis=0).reshape((1, n))

        return self

    @jit(parallel=True)
    def detect(self, x):
        """
        Calculate the p-value of a Gaussian probability density function for an observed data point in a previously given array (X in "fit" method).\n
        Args:
            x (ndarray):    an observed data point in the array X;
        Returns:
            p (ndarray):    p-value  of the observed data point;
        """
        x = x - self.mu
        pi = math.pi
        val = (-0.5) * x ** 2 / self.sigma2
        pValue = (2 * pi * self.sigma2) ** (-0.5) * np.exp(val)

        return pValue

    @jit(parallel=True)
    def update(self, x):
        """  
        Update the Gaussian fit.\n
        Args:
            x (ndarray):        an observed data point in the array X;
        Returns:
            mu (ndarray):       updated mean value of a given array;
            sigma2 (ndarray):   updated variance of the given array;
        """
        mu = self.mu
        sigma2 = self.sigma2
        m1 = self.m

        for i in range(x.shape[0]):
            m2 = m1 + 1
            sigma2 = np.multiply(
                np.divide(m1, m2), (sigma2) + np.divide((x[i] - mu) ** 2, m2)
            )
            mu = mu + np.divide((x[i] - mu), m2)

            m1 = m2

        self.m = m1
        self.mu = mu
        self.sigma2 = sigma2

        return self

    @jit
    def save(self, filename):
        """  
        Save the given file as a compressed pickle;\n
        Args:
            filename (str): path to save and the name of the file to be compressed, pickled, and saved;
        """
        pickle.dump(self, bz2.BZ2File(filename, "wb"), protocol=resCons.model.protocol)

    @jit
    def load(self, filename):
        """  
        Load the compressed pickled file, decompress it, and unpickle it;\n
        Args:
            filename (str): name and path to where the pickled compressed file is stored; 
        """
        return pickle.load(bz2.BZ2File(filename, "rb"))

    @staticmethod
    def score(trueVal, predVal):
        """  
        Get the precision and recall scores by calculating the number of true positives, false positives, and false negatives;\n
        NOTE: Not currently used;\n
        Args:
            trueVal (ndarray):  true value of the observed data point in the array X; 
            predVal (ndarray):  predicted value of the observed data point in the array X;
        Returns:
            prec (float):       precision score (between 0 and 1);
            rec (float):        recall score (between 0 and 1);
        """
        prec = 0
        rec = 0

        try:

            tp = np.where((trueVal == 1) & (predVal == 1), 1, 0).sum(axis=0)
            fp = np.where((trueVal == 0) & (predVal == 1), 1, 0).sum(axis=0)
            fn = np.where((trueVal == 1) & (predVal == 0), 1, 0).sum(axis=0)

            prec = tp / (tp + fp)
            rec = tp / (tp + fn)

        except ZeroDivisionError:
            log.warnings("No anomaly found")

        return prec, rec

    def copy(self):
        """  
        Create a deep copy of whatever object it is called for;
        """
        return deepcopy(self)
