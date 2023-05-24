import os

import numpy as np
import pandas as pd

from erroneousdetector.main.data.utils.Data import Data
from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.Constants import Resource as resCons


class Train:
    """  
    A class to process (load) the train data. 

    Available class method(s):
        get: validates and transforms train data into a matrix form;
    """

    def get(self, ts, transform=True):
        """  
        Load raw train data dataframe for a single timeseries, validate it, and transform it to be the same format as the test data 
        (matrix with IDs as columns and gasDay as index, suitable for the model).
        NOTE: To be called in a timeseries for-loop.

        Args:
            ts (str):           timeseries name to be selected;

            transform (Bool):   True/False indicating whether or not do all transformations (i.e. matrix form);
        
        Returns:
            data (df):          final form of the train dataframe (matrix form) for a single timeseries, ready to be fed to the model;
                                if transform = False: raw train dataframe for a single timeseries, in a regular, non-matrix form;           
        """

        data = Data.getData(ts, envCons.trainDataPath)
        self.__validate(data)

        if transform:
            data = self.__transform(data)

        return data

    def __validate(self, data):
        """  
        Checks if the necessary columns are present in the raw train and raises an exception if they are not.

        Args:
            data (df):  dataframes of all available raw train data for an individual timeseries;
                        
        Returns:
            Exception message or None;
        """

        areColumnsPresent = np.all(
            [
                c in data.columns
                for c in [
                    resCons.data.timeseriesColumn,
                    resCons.data.gasColumn,
                    resCons.data.netzColumn,
                    resCons.data.bilanzColumn,
                    resCons.data.gasDayColumn,
                    resCons.data.dailyValueColumn,
                ]
            ]
        )

        if not areColumnsPresent:
            raise Exception(
                "Required columns do not exist in the provided data frame, please reffer to the 'resource.yml' file."
            )

    def __transform(self, data):
        """
        Transform the training data into a format used for the model (matrix with IDs as columns and gasDay as index). 
        Create a multiIndex and the 'id' column, drop all data sets which have all values as nulls or 0s, 
        fill in the nulls and 0s in other data sets, and create the final form of the train data.

        Args: 
            data (df):              transformed dataframe for all available data, for an individual timeseries; 
                                    Df have a single "counter" index and 9 columns: 
                                    mandantColumn, wertegangColumn, netzColumn, gasColumn, bilanzColumn, 
                                    netzuebergangColumn, timeseriesColumn, gasDayColumn, and a dailyValueColumn
        Returns:  
            dataTransformed (df):   train dataframe (matrix form) for a single timeseries, ready to be fed to the model;
        """
        # data = data.copy()

        # set gas day as index
        dataTransformed = Data.createGasDayIndex(data)

        index = list(dataTransformed.index)
        # find missing values/0s and drop them
        self.dateRange = pd.date_range(min(index), max(index), freq="D")
        dataTransformed = Data.dropNullsOriginalData(dataTransformed)

        # replace missing values and 0s with the mean daily values
        dataTransformed = dataTransformed.groupby(
            resCons.data.idColumn, as_index=True, group_keys=False
        ).apply(lambda row: Data.reindexFillNullsAndZeros(row, self.dateRange))

        # change the type of daily values from int to float
        dataTransformed = dataTransformed.astype(
            {resCons.data.dailyValueColumn: np.float64}
        )

        # pivot the data to create a matrix suitable for the model
        dataTransformed = Data.pivotData(dataTransformed)

        return dataTransformed
