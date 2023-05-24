import os
from pathlib import Path

import dask.dataframe as dd
import numpy as np

from erroneousdetector.main.data.utils.Data import Data
from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.Constants import Resource as resCons
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)


class RawTest:
    """ 
    A class used to process test data.\n
    Available class methods:
        get: load, validate, add an id column, and select only the best of values from the test data; 
        getStreamlit: validate, add an id column, and select only the bestOf values from the input test data in the Streamlit app;
        transformAndSave: transforms the raw test data into dataframes for individual timeseries; 
    """

    def get(self, filePath: str = envCons.rawTestDataPath):
        """
        Load the test data from a designated folder, validate it, create an id column, and select only the Best Of data;

        Args:
            filePath (str):     path to where the test data .csv is stored;

        Returns:
            self.rawData (df):  test dataframe with only Best Of data and added id column for an individual timeseries;
        """
        # load the test data
        self.rawData = RawTest.__getRawTestData(filePath)

        # validate that all the necessary columns are present in the data
        self.__validate(self.rawData)

        return self.rawData

    def getStreamlit(self, testDF):
        """  
        Validate and select only bestOf data from an input test data file;
        This class method work when the file is uploaded through Streamlit;

        Args:
            testDF (df):    test data, all timeseries together, preliminary and bestOf in one column;

        Returns:
            testData (df):  test data, all timeseries together, only bestOf values;
        """
        self.rawData = testDF

        # validate that all the necessary columns are present in the data
        self.__validate(self.rawData)

        return self.rawData

    def transformAndSave(
        self, ts, save: bool = True, savePath: str = envCons.testDataPath
    ):
        """
        Transform the test data for all timeseries into test data for individual timeseries and save them;
        
        Args:
            ts (str):       timeseries name to be selected in a for-loop;
            save (bool):    True/False - determines whether or not to save the transformed test data;
            savePath (str): path to where to save the separated test dataframes; 
                            default is testPath;
        Returns:
            tsTestDF (df):  test data for an individual timeseries;
        """

        # transform the test data for each timeseries individually
        testDataDask = RawTest.__transformRawTestData(self.rawData, ts)
        testDataPandas = testDataDask.compute()

        log.debug(
            "Data transformed and loaded : No.of IDs = {},  Total data points {}".format(
                len(testDataPandas.id.unique()), len(testDataPandas.index)
            )
        )

        # save the transformed test data to be loaded in Test.py
        if save:
            Data.saveData(testDataPandas, ts, savePath)

        return testDataPandas

    @staticmethod
    def __getRawTestData(filePath: str):
        """  
        Load the test data previously put in the designated folder;

        Args:
            filePath (str): path to where the test data csv is stored;

        Returns:
            testDF (df):    test data for an individual timeseries;
        """

        if not any(Path(filePath).glob("*{}".format(resCons.model.csvFileExtn))):
            raise Exception(
                "Test folder '{}' does not contains any (.csv) files".format(filePath)
            )

        testDF = dd.read_csv(
            os.path.join(filePath, "*{}".format(resCons.model.csvFileExtn)),
            parse_dates=[resCons.data.gasDayColumn],
            dayfirst=True,
        )

        log.trace(
            "Original data loaded : {} => {}".format(testDF.columns, testDF.index)
        )

        # select only the bestOf data for testing
        testDF = testDF[
            testDF[resCons.data.preliminaryOrBestOfColumn]
            == int(envCons.testQualityCode)
        ]

        return testDF

    def __validate(self, testDF):  # with added some cleanup
        """  
        Check if the necessary columns are present in the test data and raises an exception if they are not.

        Args:
            testDF (df):   test data, all timeseries together; 

        Returns:
            Exception message or None;
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
                "Required columns do not exist in the provided data frame, please refer to the 'resource.yml' file."
            )

    @staticmethod
    def __transformRawTestData(originalData, ts):
        """  
        Transform the test data for all timeseries to test data for individual timeseries;

        Args:
            originalDict (dict):    test data for all timeseries together;
            ts (str):               timeseries name to be selected via for-loop;

        Returns:
            tsTestDF (df):         transformed dataframe for all available test data, for an individual timeseries; 
        """
        tsTestDF = originalData[originalData[resCons.data.timeseriesColumn] == ts]

        log.trace(
            "Timeseries data is loaded : {} = {}".format(
                tsTestDF.columns, len(tsTestDF.index)
            )
        )

        # changes the type from int to float, so it matches with train data
        tsTestDF = tsTestDF.astype({resCons.data.dailyValueColumn: np.float64})

        # create the id column from gas type, netz, and bilanz columns
        tsTestDF = Data.createIdColumn(tsTestDF)

        return tsTestDF
