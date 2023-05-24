import os

import numpy as np
import pandas as pd

from erroneousdetector.main.data.utils.Data import Data
from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.Constants import Resource as resCons
import shutil


class Test:
    """  
    A class to process the already separated daily data. 

    Available class methods:
        get: validates and transforms test data into a matrix form;

        mergeCorrectedAndRawTestData: merge the raw test data with the corrected test data and transform it to be saved as the result;

    NOTE: All available class methods are to be called in a timeseries for-loop;
    """

    def get(self, ts, transform=True):
        """  
        Load test data dataframe for a single timeseries, validate it, and transform it to be the same format as the train data 
        (matrix with IDs as columns and gasDay as index, suitable for the model).
        NOTE: To be called in a timeseries for-loop;

        Args:
            ts (str):           timeseries name to be selected;

            transform (Bool):   True/False indicating whether or not do all transformations (i.e. matrix form)
        
        Returns:
            data (df):          test dataframe (matrix form) for a single timeseries, ready to be fed to the model;
                                if transform = False: raw test dataframe for a single timeseries, in a regular, non-matrix form;           
        """
        # load the test data separated by timeseries
        data = Data.getData(ts, envCons.testDataPath)

        # validate the test data
        self.__validate(data)

        # transform the test data
        if transform:
            data = self.__transform(data)

        return data

    def mergeCorrectedAndRawTestData(self, raw, corrected):
        """
        Merge the raw test data with the corrected test data for a single timeseries and transform it so that it looks exactly like test data;
        NOTE: To be called in a timeseries for-loop;

        Args:
            raw (df):                   raw test data for a single timeseries with no transformation done;

            corrected (df):             dataframe of corrected daily values in the "TAGESWERT_CORRECTED" column;

        Returns:
            originalAndCorrected (df):  test data with the appended corrected values in the "TAGESWERT_CORRECTED" column;
        """

        corrected[resCons.data.gasDayColumn] = corrected.index
        ## un-pivot the corrected df to have the same form as test and train data
        corrected = corrected.melt(
            id_vars=[resCons.data.gasDayColumn],
            var_name=resCons.data.idColumn,
            value_name=resCons.data.correctedColumn,
        )

        # round up the corrected value to have only one decimal point
        corrected[resCons.data.correctedColumn] = round(
            corrected[resCons.data.correctedColumn], 0
        )

        # add the corrected data to the original test data (rawTest)
        originalAndCorrected = pd.merge(
            raw,
            corrected,
            on=[resCons.data.idColumn, resCons.data.gasDayColumn],
            how="left",
        )

        # in case the test data was missing or was 0 for a given date, ignore the correction and keep null/0 value
        # in case the corrected data is null
        originalAndCorrected[resCons.data.correctedColumn] = np.where(
            (originalAndCorrected[resCons.data.dailyValueColumn].isna())
            | (originalAndCorrected[resCons.data.dailyValueColumn] == 0)
            | (originalAndCorrected[resCons.data.correctedColumn].isna()),
            originalAndCorrected[resCons.data.dailyValueColumn],
            originalAndCorrected[resCons.data.correctedColumn],
        )

        # keep only data points which were corrected
        originalAndCorrected = originalAndCorrected[
            (originalAndCorrected[resCons.data.dailyValueColumn])
            != (originalAndCorrected[resCons.data.correctedColumn])
        ]

        # drop all the unnecessary columns
        originalAndCorrected.drop(
            columns=[resCons.data.dailyValueColumn, resCons.data.idColumn,],
            axis=1,
            inplace=True,
        )

        # rename the id and correctedValue columns
        originalAndCorrected.rename(
            columns={resCons.data.correctedColumn: resCons.data.dailyValueColumn,},
            inplace=True,
        )

        # make all  number columns integers
        # originalAndCorrected[resCons.data.numericColumns] = originalAndCorrected[resCons.data.numericColumns].astype(int)

        return originalAndCorrected

    def __validate(self, testDF):
        """  
        Checks if the necessary columns are present in the sorted test data and raises an exception if they are not.

        Args:
            testDF (df):   dataframe for a single timeseries test data

        Returns:
            Exception message or None
        """

        areColumnsPresent = np.all(
            [
                c in testDF.columns
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
        Transform single timeseries data to a matrix form suitable for the model.

        Args:
            data (df):              dataframe for a single timeseries test data;

        Returns:
            dataTransformed (df):   dataframe for a single timeseries test data with null/0 values dropped and with IDs as columns and gasDay as index;
        """
        # create multiIndex and 'id' column
        dataTransformed = Data.createGasDayIndex(data)

        # find missing values/0s and drop them
        self.dateRange = pd.date_range(
            dataTransformed.index.min(), dataTransformed.index.max(), freq="D"
        )
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
