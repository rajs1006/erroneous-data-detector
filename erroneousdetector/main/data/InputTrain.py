import glob
import os
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd

from erroneousdetector.main.data.utils.Data import Data
from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.Constants import Resource as resCons
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)


class RawTrain:
    """  
    A class used in the calibration step to transform the raw train data based on timeseries;\n
    NOTE: To be called in the CALIBRATION step;\n
    Available class methods:
        get: load and validate all of the available train data;
        transformAndSave: transform the raw train data into train data for individual timeseries; 
    """

    def get(self, rawPath: str = envCons.rawTrainDataPath):
        """  
        Load all of the available train data into a dictionary, with months as keys and validate it.
        Args:
            rawPath (str):  path to where the raw data is stored;
        Returns:
            rawData (df):   train data, all timeseries together; 
        """
        # load the raw train monthly data given file path
        self.rawData = RawTrain.__getRawTrainData(rawPath)

        # validate that all the necessary columns are present in the data
        self.__validate(self.rawData)

        return self.rawData

    def transformAndSave(
        self, ts, save: bool = True, savePath: str = envCons.trainDataPath
    ):
        """  
        Transform train data by timeseries;
        
        Args:
            ts (str):       timeseries name to be selected in a for-loop;
            save (bool):    True/False - determines whether or not to save the transformed train data;
            savePath (str): pass the path to the location where to save the train data;

        Returns:
            trainDF (df):   transformed dataframe for all available train data, for an individual timeseries;
        """
        # transform the train data for each timeseries individually
        trainDataDask = RawTrain.__transformRawTrainData(self.rawData, ts)
        trainDataPandas = trainDataDask.compute()

        log.debug(
            "Data transformed and loaded : No.of IDs = {},  Total data points {}".format(
                len(trainDataPandas.id.unique()), len(trainDataPandas.index)
            )
        )

        # save the transformed train data to be loaded in Train.py
        if save:
            Data.saveData(trainDataPandas, ts, savePath)

        return trainDataPandas

    @staticmethod
    def __getRawTrainData(filePath: str):
        """
        Handle general loading of the original raw data into a dictionary of monthly dataframes;
        File extension should be .csv;

        Args: 
            filePath (str): path to where the raw train data is stored;

        Returns:
            trainDF (df):   train data, all timeseries together;
        """
        if not any(Path(filePath).glob("*{}".format(resCons.model.csvFileExtn))):
            raise Exception(
                "Train folder '{}' does not contains any (.csv) files".format(filePath)
            )
        
        # load the raw monthly data as it is
        trainDF = dd.read_csv(
            os.path.join(filePath, "*{}".format(resCons.model.csvFileExtn)),
            parse_dates=[resCons.data.gasDayColumn],
            dayfirst=True,
        )

        log.trace(
            "Original data loaded : {} => {}".format(trainDF.columns, trainDF.index)
        )

        # select only the bestOf data for training
        trainDF = trainDF[
            trainDF[resCons.data.preliminaryOrBestOfColumn]
            == int(envCons.trainQualityCode)
        ]

        return trainDF

    def __validate(self, rawTrain):
        """  
        Check if the necessary columns are present in the train data and raise an exception if they are not;

        Args:
            rawTrain (df):  dataframes of raw monthly data, all timeseries together;

        Returns:
            Exception message or None;
        """
        areColumnsPresent = np.all(
            [
                c in rawTrain.columns
                for c in [
                    resCons.data.timeseriesColumn,
                    resCons.data.gasColumn,
                    resCons.data.netzColumn,
                    resCons.data.bilanzColumn,
                    resCons.data.preliminaryOrBestOfColumn,
                    resCons.data.gasDayColumn,
                    resCons.data.dailyValueColumn,
                ]
            ]
        )

        if not areColumnsPresent:
            raise Exception(
                "Required columns do not exist in the provided data frame, please reffer to the 'resource.yml' file, section ORIGINAL."
            )

    @staticmethod
    def __transformRawTrainData(originalData, ts):
        """  
        Transform the original raw monthly data for all timeseries into yearly data for individual timeseries;

        Args:
            originalData (df):  train data, all timeseries together;
            ts (str):           timeseries name to be selected via for-loop in the File.py

        Returns:
            tsTrainDF (df):     transformed dataframe for all available train data, for an individual timeseries;
        """

        # create an empty dataframe for an individual timeseries

        # loop through monthly dataframes and populate the dataframe for each timeseries
        # for month in originalDict.values():
        tsTrainDF = originalData.loc[originalData[resCons.data.timeseriesColumn] == ts]

        log.trace(
            "Timeseries data is loaded : {} => {}".format(
                tsTrainDF.columns, len(tsTrainDF.index)
            )
        )

        # creates the id column
        tsTrainDF = Data.createIdColumn(tsTrainDF)

        return tsTrainDF
