import bz2
import pickle
from copy import deepcopy

import numpy as np
from numba import jit
from sklearn.preprocessing import PowerTransformer, StandardScaler

from erroneousdetector.main.utils.Constants import Resource as resCons
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)


class Transform:
    """  
    A class to encapsulate methods needed for the data transformation via standard scaler or power transformation (Yeo-Johnson).\n
    Available class methods:
        fit: fit the data with a chosen transformer;
        partial_fit: fit the data with a chosen transformer;
        transform: center and standardize the data based on the chosen transformer;
        fit_transform: 
        inverse_transform: inverse transform the data;
        update_mean: update the mean of the data;
        save: compress, pickle, and save a file;
        load: load, decompress, and unpickle a file;
        deepcopy: create a deep copy of an object;
    """

    def __init__(self, standardize=False):

        self.mean = 0
        self.scaler = StandardScaler(copy=False)

    def fit(self, data):
        """  
        Fit the data with the chosen transformer;\n
        Args:
            data (ndarray): input array - train or test data;
        """
        try:
            self.scaler.fit(data)
        except Exception as e:
            log.error("Tranformations 'FIT' failed for {} ==>> {}".format(data, e))

        return self

    def partial_fit(self, data):
        """ 
        Fit the data with the chosen transformer- partial fit is there in case the data is too large;\n
        Args:
            data (ndarray): input array - train or test data; 
        """
        try:
            self.scaler.partial_fit(data)
        except Exception as e:
            log.error(
                "Tranformations 'PARTIAL_FIT' failed for {} ==>> {}".format(data, e)
            )

        return self

    def transform(self, data):
        """  
        Center and scale the data based on the chosen transformer;\n
        Args:
            data (ndarray): input array to transform - train or test data;
        Returns:
            data (ndarray): transformed input data;
        """
        try:
            transformedData = self.scaler.transform(data)
            return transformedData
        except Exception as e:
            log.error(
                "Tranformations 'TRANSFORM' failed for {} ==>> {}".format(data, e)
            )

        return data

    def fit_transform(self, data):
        """  
        Fit to data and transform it with the chosen transformer;\n
        Args:
            data (ndarray): input data - train or test data;
        Returns:
            data (ndarray): transformed input data;
        """
        try:
            transformedData = self.scaler.fit_transform(data)
            return transformedData
        except Exception as e:
            log.error(
                "Tranformations 'FIT_TRANSFORM' failed for {} ==>> {}".format(data, e)
            )

        return data

    def inverse_transform(self, data):
        """  
        Inverse transform the data - back to the original values;\n
        Args:
            data (ndarray): input array transformed with the chosen transformer;
        Returns:
            data (ndarray): input array reverted back to the actual values;
        """
        try:
            transformedData = self.scaler.inverse_transform(data)
            return transformedData
        except Exception as e:
            log.error(
                "Tranformations 'INVERSE_TRANSFORM' failed for {} ==>> {}".format(
                    data, e
                )
            )

        return data

    @jit(parallel=True)
    def updateMean(self, data, oldMean, l):
        """  
        Update the mean of the data;\n
        Args:
            data (ndarray):     input array (test data);
            oldMean (float):    old mean of the input array (test data);
            l (int):            length of the input array (test data);
        Returns:
            mu (float):         new mean of the input array;
        """
        mu = oldMean
        m1 = l

        for i in range(data.shape[0]):
            d = data[i]
            if d != 0 and not np.isnan(d):
                m2 = m1 + 1
                mu = mu + np.divide((d - mu), m2)
                m1 = m2

        return mu

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

    def deepcopy(self):
        """
        Create a deep copy of whatever object it is called for;
        """
        return deepcopy(self)
