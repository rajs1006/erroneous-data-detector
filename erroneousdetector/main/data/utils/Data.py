import bz2
import os
import pickle
import shutil
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.Constants import Resource as resCons
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)


class Data:
    """  
    Encapsulates the utility functions used for dealing with data: input and transformation/preparation.

    Available class methods:
        createIdColumn: create the id column from gas, netz, and bilanz columns;         
        createGasDayIndex: set gas day as index column;
        getData: load the saved pickled data into a dataframe;
        dropNullsOriginalData: remove all data sets which have all values as 0 or missing;
        pivotData: transform data into a matrix form suitable for the model;
        saveData: save data as a compressed pickle;
        saveResult: create a result .csv file;
        saveHistory: save a .txt file with info on last calibration, training, and testing;
        moveToArchive: move old result/test file to the archive folder;
        reindexFillNullsAndZeros: replace missing values with 0s;
    """

    @staticmethod
    def createIdColumn(data):
        """  
        Create the id column from gasColumn, netzColumn, and bilanzColumn; \
        
        Args:
            data (df):  df we want to append the column to;   

        Return:
            data (df):  same df with the id column appended
        """

        data[resCons.data.idColumn] = (
            data[resCons.data.groupByColumns[0]].astype(str)
            + "_"
            + data[resCons.data.groupByColumns[1]].astype(str)
            + "_"
            + data[resCons.data.groupByColumns[2]].astype(str)
        )

        return data

    @staticmethod
    def createGasDayIndex(data):
        """  
        Replace the existing index with the gasDayColumn as datetime index;

        Args:
            data (df):  df we want to set the gas day as index to; 

        Return:
            data(df):   same df with the gas day as index column;
        """

        data.reset_index(drop=False, inplace=True)
        data.set_index(resCons.data.gasDayColumn, drop=True, inplace=True)

        return data

    @staticmethod
    def getData(ts, filePath: str):
        """ 
        Handles general loading of the saved compressed pickled data into a dataframe;
        
        Args:
            ts (str):       timeseries name to be selected via for-loop; 
            filePath (str): path to where the data is stored;

        Returns:
            df (df):        loaded dataframe; 
        """
        if not any(Path(filePath).glob("*{}".format(resCons.model.dataFileExtn))):
            raise Exception(
                "@folder '{}' : no (.data) files are to be processed, try again with '--calibrate' option".format(
                    filePath
                )
            )

        df = pd.read_pickle(
            os.path.join(filePath, "{}{}".format(str(ts), resCons.model.dataFileExtn)),
            compression="bz2",
        )

        return df

    @staticmethod
    def dropNullsOriginalData(data):
        """
        Remove all data sets for which all daily values are missing (null) or have 0 values.
        
        Args:
            data (df):  dataframe for an individual timeseries;

        Returns: 
            data (df):  dataframe without data sets which has all daily values missing or 0;
        """

        data = data.groupby(
            resCons.data.idColumn, as_index=False, group_keys=False
        ).filter(
            lambda x: ~(
                (
                    (x[resCons.data.dailyValueColumn].isna().all())
                    | ((x[resCons.data.dailyValueColumn] == 0).all())
                )
            )
        )

        return data

    @staticmethod
    def pivotData(data):
        """ 
        Create a matrix for an individual timeseries to be used in the model;
        
        Args:
            data (df):  dataframe for an individual timeseries, with IDs as values in the "id" column;

        Returns:
            data (df): dataframe for an individual timeseries, with each ID as a separate column and "GASTAG" as index;
        """
        try:
            data[resCons.data.gasDayColumn] = data.index
            data = data.pivot(
                index=resCons.data.gasDayColumn,
                columns=resCons.data.idColumn,
                values=resCons.data.dailyValueColumn,
            )
        except:
            raise Exception(
                "Timeseries contains duplicate combination of \
                        (GASTAG, ID_BKN_NETZ, ID_BKN_BILANZOBJ, ID_BKN_MARKTGEBIET, BKN_MENGENTYP	, BKN_QUALITAET) "
            )

        return data

    @staticmethod
    def saveData(df, ts, savePath: str):
        """
        Save dataframe for individual timeseries as a compressed pickle;
        
        Args:
            df (df):        df to be pickled, compressed, and saved;
            ts (str):       timeseries name to be selected via for-loop;
            savePath (str): path to where the data is to be saved;
        """

        df.to_pickle(
            os.path.join(savePath, "{}{}".format(ts, resCons.model.dataFileExtn)),
            compression="bz2",
            protocol=resCons.model.protocol,
        )

    @staticmethod
    def saveResult(df, savePath: str = envCons.saveResultPath):
        """ 
        Save the corrected data in the results folder. Append if there's already the file for the day, else write a new one.
        
        Args:
            df (df):        result data - only corrected values;
            savePath (str): path to where to save the corrected data; 
                            default is data/result; 
        """
        fileName = os.path.join(
            savePath,
            "{}{}".format(
                str(date.today().strftime("%d-%m-%Y")), resCons.model.csvFileExtn
            ),
        )

        if os.path.exists(fileName):
            df.to_csv(fileName, mode="a", header=False, index=False)
        else:
            df.to_csv(fileName, mode="w", index=False)

    @staticmethod
    def saveHistory(
        ts=None,
        start=False,
        calib=False,
        train=None,
        test=None,
        fileName: str = envCons.historyFilePath,
    ):
        """ 
        Save the calibration, training, or testing info. Append if there's already a file, else write a new one.
        
        Args:
            ts (str):       timeseries name;
            start (bool):   True if we want to log the start of the running of the app;
            calib (bool):   True if we want to save the date and time when calibration was done;
            train (df):     train data;
            test (df):      test data;
            savePath (str): path to where to save the corrected data; 
                            default is data/result; 
        """
        currentTimestamp = datetime.now()

        f = open(fileName, "a+")

        if start:
            f.write("{}: APP STARTED \n".format(currentTimestamp))

        if calib:
            f.write(
                "{}: Calibration for {} timeseries was done.".format(
                    currentTimestamp, ts
                )
            )

        if train is not None:
            trainDayMax = train.index.date.max()
            trainDayMin = train.index.date.min()
            f.write(
                "\n{}: Training for {} timeseries was done from {} to {}.".format(
                    currentTimestamp, ts, trainDayMin, trainDayMax,
                )
            )

        if test is not None:
            testDayMax = test.index.date.max()
            testDayMin = test.index.date.min()

            if test.index.date.min() == test.index.date.max():
                f.write(
                    "\n{}: Testing for {} timeseries was done for {}.".format(
                        currentTimestamp, ts, testDayMax
                    )
                )
            else:
                f.write(
                    "\n{}: Testing for {} timeseries was done from {} to {} \n.".format(
                        currentTimestamp, ts, testDayMin, testDayMax,
                    )
                )

            if envCons.saveModelWhileTest:
                f.write(
                    "\n{}: Models for {} timeseries are updated until {}.\n\n".format(
                        currentTimestamp, ts, testDayMax
                    )
                )

        f.close()

    @staticmethod
    def moveToArchive(test=False, result=False):
        """  
        Move the old results and test files into archive folder.

        Args:
            test (bool):    choose if we want to move the test to the archive/test;
            result (bool):  choose if we want to move the result to the archive/result;
        """
        if envCons.archiveStrategy == 1:
            if test:
                source = envCons.rawTestDataPath
                destination = os.path.join(
                    envCons.archiveTestPath,
                    str(datetime.today().strftime("%d%m%Y_%H%M%S")),
                )
                os.makedirs(destination, exist_ok=True)

            if result:
                source = envCons.saveResultPath
                destination = envCons.archiveResultPath

            for file in os.listdir(source):
                if file.endswith(resCons.model.csvFileExtn):
                    try:
                        shutil.move(os.path.join(source, file), destination)
                    except:
                        log.warning(
                            "No file with name {} at location {} to move".format(
                                file, source
                            )
                        )

        elif envCons.archiveStrategy == 0:
            pass

    @staticmethod
    def reindexFillNullsAndZeros(data, dateRange):
        """  
        Find and replace all missing values with 0s for each data set in a timeseries and fill other columns with previous/next values.

        Args:
            data (df):                  dataframe with missing values/0s present;
            dateRange (DatetimeIndex):  date-time index which spans from the first to the last available day, 
                                        filling in the missing days;
                                        
        Returns:
            data (df):                  dataframe with no missing values;
        """

        # reindex data
        data = data.reindex(dateRange)

        # replace 0 with null, so that it can be filled to avoid the bfill, ffill
        # data[resCons.data.dailyValueColumn] = data[
        #     resCons.data.dailyValueColumn
        # ].fillna(0)

        # fill nulls for the other columns
        data.fillna(method="bfill", inplace=True)
        data.fillna(method="ffill", inplace=True)

        return data
