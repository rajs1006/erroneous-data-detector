import bz2
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from numba import jit
from pmdarima import auto_arima

from erroneousdetector.main.utils.Constants import Resource as resCons
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)


class AutoArima:
    """  
    A class to encapsulate all methods for value prediction using auto_arima package;\n
    Available class methods:\n
        build: build the auto_arima model for a data set;
        predict: forecast the value based on the auto_arima model from "build";
        update: update the arima prediction model;
        save: compress, pickle, and save a file;
        load: load, decompress, and unpickle a file;
        getResidual: get the residual of the auto_arima model from "build";
        copy: create a deep copy of an object;
    """

    def __init__(self):
        """  
        The constructor adds the following class attributes: 
            trainArgs (dict):   dict of parameters required by auto_arima; includes the range for p and q ARIMA parameters, 
                                max. seasonality parameter,regular and seasonal differencing parameter, etc; 
                                Full documentation can be found at: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html 
            testArgs (dict):    specify if we want to get the confidence interval or not at the output of the prediction;
        """
        self.trainArgs = dict(
            start_p=0,
            start_q=0,
            max_p=2,
            max_q=2,
            m=5,
            seasonal=True,
            d=0,
            D=0,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        self.testArgs = dict(return_conf_int=True)

    def build(self, data, exog=None):
        """  
        Build the auto_arima model for a particular data set and exogenous variables if provided;\n
        Args:
            data (ndarray): data set to train the auto_arima on;
            exog (ndarray): exogenous variables to be used as additional features in the regression operation;
        Returns:
            model (object): a defined auto_arima model - to be used with an auto_arima internal methods;
        """
        # add some random number to avoid singular matrix
        data = data.ravel()

        try:
            if exog is not None:
                self.trainArgs["exogenous"] = exog

            self.model = auto_arima(data, **self.trainArgs)
        except Exception as e:
            log.warning("Error : Build failed : {} ==>> {}".format(data, e))

        return self

    def predict(self, exog=None, steps=1, return_conf_int=False, alpha=0.05):
        """
        Forecast the the values using the defined auto_arima model;\n
        Args:
            exog (ndarray):                 exogenous variables to be used as additional features in the regression operation;
            steps (int):                    number of periods in the future to forecats;
            return_conf_int (Bool):         whether to get the confidence intervals of the forecasts; default is False;  
            alpha (float):                  confidence interval parameter - it is (1-alpha)%;
        Returns:
            forecast (ndarray):             the array of forecasted value(s);
            confidenceInterval (ndarray):   the confidence interval for the forecast; 
        """
        if exog is not None:
            self.testArgs["exogenous"] = exog
            self.testArgs["alpha"] = alpha

        self.forecast, self.confidenceInterval = self.model.predict(
            steps, **self.testArgs
        )

        return self.forecast, self.confidenceInterval

    def update(self, data, exog=None):
        """  
        Update the auto arima model.\n
        Args:
            data (ndarray): data set to train the auto_arima on;
            exog (ndarray): exogenous variables to be used as additional features in the regression operation;
        """
        try:
            # add some random number to avoid singular matrix
            data = data.ravel()
            self.model.update(data, exogenous=exog)
        except Exception as e:

            log.warning(
                "Warning :  Arima update failed for : {} ==>> {}".format(data, e)
            )

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

    def getResidual(self):
        """  
        Get the residual of the defined auto_arima model;
        """
        return self.model.resid()

    def copy(self):
        """  
        Create a deep copy of whatever object it is called for;
        """
        return deepcopy(self)
