### needed libraries:
import itertools
import os
import time
from datetime import datetime

import dask.dataframe as ddf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as py
from joblib import Parallel, delayed
from numba import jit


### needed files:
from erroneousdetector.main.data.Test import Test
from erroneousdetector.main.data.Train import Train
from erroneousdetector.main.models import AutoArima, Gaussian, Transformer
from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.Constants import Resource as resCons
from erroneousdetector.main.utils.decorator.Log import time_log_this
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)


class Model:
    """  
    Available class methods:
        trainAndValidate: train and validate the model;
        detectAndCorrect: run the detection and prediction on the test data;
        plot: plot the train and test data with the corrected test values added;     
    """

    def __call__(self, timeseries: str, parallel: bool = True):
        """ 
        The __call__ method creates the following class attributes:
            timeseries (str):   timeseries to work with;
            parallel (bool):    Whether or not to run the parallel processing; 
        """
        ## Create directory for every timeseries in Model
        os.makedirs(os.path.join(envCons.modelPath, timeseries), exist_ok=True)
        self.modelPath = os.path.join(envCons.modelPath, timeseries)

        self.fig = go.Figure()

        self.pt = Transformer.Transform(standardize=False)
        self.predictor = AutoArima.AutoArima()
        self.detector = Gaussian.Gaussian(envCons.epsilon)

        self.timeseries = timeseries

        return self

    ### Public methods ###

    @time_log_this
    def trainAndValidate(self, train):
        """  
        Train and validate the model with the available train data; \n
        Args:
            train (df): training data for a single timeseries;
        """
        log.info(
            "Timestamp of train data: {} -> {}".format(
                train.index[0].strftime("%d-%m-%Y"),
                train.index[-1].strftime("%d-%m-%Y"),
            )
        )

        ## Load the dataframe with Dask, as it is a faster alternative to pandas
        # transform the dask df to pandas df in the end with "compute"
        df = ddf.from_pandas(train.T, npartitions=envCons.coreProcess)
        df.map_partitions(
            Model.__trainValidate,
            self.pt.deepcopy(),
            self.predictor.copy(),
            self.detector.copy(),
            self.modelPath,
            meta=("str"),
        ).compute(scheduler=envCons.scheduler)

    @time_log_this
    def detectAndCorrect(self, test):
        """  
        Run the detection and preditcion model on the test data;\n
        Args:
            test (df):      test data to run the model son;
        Returns:
            corrected (df): corrected data;  
        """
        log.info(
            "Timestamp of train data: {} -> {}".format(
                test.index[0].strftime("%d-%m-%Y"), test.index[-1].strftime("%d-%m-%Y")
            )
        )

        df = ddf.from_pandas(test.T, npartitions=envCons.coreProcess)
        corrected = df.map_partitions(
            Model.__detectAndCorrect,
            self.pt.deepcopy(),
            self.predictor.copy(),
            self.detector.copy(),
            self.modelPath,
            meta=("str"),
        ).compute(scheduler=envCons.scheduler)

        ## Convert list of series to DataFrame
        corrected = pd.concat(itertools.chain(*corrected), axis=1)
        return corrected

    def plot(self, ts, train, test, corrected):
        """  
        Plot the test data along with the corrected test data;\n
        NOTE: it is not being called in the main app;\n
        Args:
            test (df):      test data;
            corrected (df): corrected test data;
        """
        trainFalse = []
        trainTrue = []
        tTrue = []
        testFalse = []
        correctedFalse = []

        l = 1

        if test is not None and train is not None:
            for cl in test.columns:

                self.fig.add_trace(
                    go.Scatter(
                        x=train.index,
                        y=train[cl] if cl in train.columns else [],
                        mode="lines+markers",
                        marker_color="#283C5A",
                        marker_size=4,
                        name="Train->" + cl,
                        legendgroup=str(l),
                    )
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=test.index,
                        y=test[cl],
                        mode="lines+markers",
                        marker_color="green",
                        marker_size=4,
                        name="Test->" + cl,
                        legendgroup=str(l),
                    )
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=test.index,
                        y=corrected[cl],
                        mode="lines+markers",
                        marker_color="blue",
                        marker_size=4,
                        name="Corrected->" + cl,
                        legendgroup=str(l),
                    )
                )

                trainFalse.extend([False, True, True])
                testFalse.extend([True, False, True])
                correctedFalse.extend([True, True, False])

                l = l + 1
        elif test is not None:
            for cl in test.columns:

                self.fig.add_trace(
                    go.Scatter(
                        x=test.index,
                        y=test[cl],
                        mode="lines+markers",
                        marker_color="green",
                        marker_size=4,
                        name="Test->" + cl,
                        legendgroup=str(l),
                    )
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=test.index,
                        y=corrected[cl],
                        mode="lines+markers",
                        marker_color="blue",
                        marker_size=4,
                        name="Corrected->" + cl,
                        legendgroup=str(l),
                    )
                )

                testFalse.extend([False, True])
                correctedFalse.extend([True, False])

                l = l + 1

        elif train is not None:
            if cl in train.columns:
                self.fig.add_trace(
                    go.Scatter(
                        x=train.index,
                        y=train[cl],
                        mode="lines+markers",
                        marker_color="#283C5A",
                        marker_size=4,
                        name="Train->" + cl,
                        legendgroup=str(l),
                    )
                )

                trainFalse.append(False)

        buttons = []
        if len(trainFalse) != 0:

            buttons.append(
                dict(
                    label="Hide Train",
                    method="restyle",
                    args=[{"visible": trainFalse}],
                )
            )

        buttons.extend(
            [
                dict(
                    label="Hide Test", method="restyle", args=[{"visible": testFalse}]
                ),
                dict(
                    label="Hide corrected",
                    method="restyle",
                    args=[{"visible": correctedFalse}],
                ),
                dict(label="Show All", method="restyle", args=[{"visible": True}]),
            ]
        )

        self.fig.update_layout(
            title="Erroneous Data Detector : {}".format(ts),
            xaxis_title="Days",
            yaxis_title="Tageswert",
            legend_title="GAS-NETZ-BILANZ : ",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.5,
                    xanchor="left",
                    y=1.1,
                    yanchor="top",
                    showactive=True,
                    buttons=buttons,
                )
            ],
        )

        py.plot(self.fig)

    ### static methods ###

    @staticmethod
    def __trainValidate(train, pt, predictor, detector, modelPath):
        """  
        Train and validate the model;\n
        Args:
            train (df):         training data for a single timeseries;
            pt (object):        class containing the transformation;
            predictor (object): an instance of the AutoArima class;
            detector (object):  an instance of the Gaussian class;
            modelPath (str):    path to where the model is to be stored;
        """

        log.debug("{} : start parallel {}".format("trainValidate".upper(), train.index))

        Parallel(n_jobs=envCons.coreProcess)(
            delayed(Model.__trainValidateParallel)(
                i, columnName, columnData, modelPath, pt, predictor, detector
            )
            for i, (columnName, columnData) in enumerate(train.T.iteritems())
        )

    @staticmethod
    # @jit(parallel=True)
    def __trainValidateParallel(
        i, columnName, columnData, modelPath, pTransformer, predictor, detector
    ):
        """  
        Train and validate the model using the parallelization method for utilizing the machine power the most and speed up the run time;\n
        Args:
            i (int):            parameter related to parallelization;
            columnName (str):   IDs of the train data - unique for a timeseries;
            columnData (df):    train data;
            modelPath (str):    path to where the model is to be stored; 
            predictor (object): an instance of the AutoArima class;
            detector (object):  an instance of the Gaussian class;
        """
        try:
            st = datetime.now()
            log.trace("Iter = {} : Start = '{}' ".format(i, columnName))

            modelFile = os.path.join(modelPath, columnName)

            trainData = (
                columnData.to_numpy().reshape((-1, 1)).astype(np.float64, copy=False)
            )
            l = len(trainData)

            ## Get the mean of train data, excluding the NaN values and fill NaN values in train data with that mean
            trainDataMean = np.nanmean(trainData)
            trainData = np.where(np.isnan(trainData), trainDataMean, trainData)

            ## Save the train data mean in transformer to be used for test data
            pTransformer.mean = trainDataMean
            ## Transform the data before training
            transformedTrainData, pt = Model.__transform(pTransformer, trainData)

            ## Take care of whole logic of detecting and validating, by calling the __detector method
            finalDetector, finalWindowSize = Model.__detector(
                detector, transformedTrainData, l,
            )

            ## Build AutoArima model; this is slow as it also performs a grid search
            log.trace(
                "Prediction data : {}".format(
                    transformedTrainData[l - finalWindowSize :]
                )
            )
            predict = predictor.build(transformedTrainData[l - finalWindowSize :])

            ## Save files, Save the scalar
            any(
                map(
                    Model.__saveModels,
                    [pt, finalDetector, predict],
                    [
                        (modelFile + resCons.model.scalarFileExt),
                        (
                            modelFile
                            + resCons.model.detectorFile
                            + resCons.model.modelFileExt
                        ),
                        (
                            modelFile
                            + resCons.model.predictorFile
                            + resCons.model.modelFileExt
                        ),
                    ],
                )
            )

            log.trace("Train data mean for {} = {}".format(columnName, trainDataMean))

            log.trace(
                "Iter = {} : End = '{}' : Elapsed time = {}".format(
                    i, columnName, (datetime.now() - st)
                )
            )

            ## Free memory for garbage collection
            pt = None

            predict = None

            finalDetector = None
            detector = None
            predictor = None

        except Exception as e:
            log.error("Training failed for columnName: {}, e: {}".format(columnName, e))

    @staticmethod
    def __detector(detector, trainData, l):
        """ Obtain the final detector model and the final window size. """
        ## Start checking from max window size and then keep decreasing
        windowSize = np.arange(
            envCons.startWindowSize, envCons.maxWindowSize, envCons.windowStride
        )

        detectorMap = map(
            Model.__detectTrain, [(detector, trainData, l, w) for w in windowSize],
        )

        ## Save the final model and params
        finalPValue, finalStd, finalWindowSize, finalDetector = max(
            list(detectorMap), key=lambda x: 0 if np.isnan(x[0]) else x[0]
        )
        log.trace(
            "Detector finished : finalPValue = {} , finalStd = {}, finalDetectorPValue = {}, finalWindowSize = {}".format(
                finalPValue, finalStd, finalDetector.eps, finalWindowSize
            )
        )
        ## Replace the pValue with epsilon if pValue is larger than that, assign it to object
        # and save the object
        finalPValue, finalStd = Model.__epsilonAndStd(finalPValue, finalStd)

        finalDetector.eps = finalPValue
        finalDetector.std = finalStd

        return finalDetector, finalWindowSize

    @staticmethod
    @jit(parallel=True)
    def __detectTrain(values):
        """ Obtain the pValue, std, window size, and detection model from the train data. """
        detector, trainData, l, w = values
        ## Select a window data
        data = trainData[l - w - 1 :]
        ## Last data point data for validation
        vSlice = 1

        ## Fit on train slice and detect on validation slice
        detect = detector.fit(data[:-vSlice])
        ## Choose the minimum pValue as that reflects that, that value is
        # the maximum of pValue it can go down to
        pValue = np.nanmin(detect.detect(data[-vSlice:]))

        ## Get the maximum std from the train data, this is to include
        # the maximum variance when testing the new data
        std = abs(max(abs(data)) - detect.mu) / np.sqrt(detect.sigma2)

        return pValue, std, w, detect

    @staticmethod
    def __detectAndCorrect(test, pt, predictor, detector, modelPath):
        """  
        Run the model on test data;\n
        Args:
            test (ndarray):     test data to run the model on;
            pt (object):        instance of the Transformer class;
            predictor (object): an instance of the AutoArima class;
            detector (object):  an instance of the Gaussian class;
            modelPath (str):    path to where the model is stored;
        Returns:
            corrected (list):   correct values for anomalous data points;
        """
        log.debug(
            "{} : start parallel {}".format("detectAndCorrect".upper(), test.index)
        )

        corrected = Parallel(n_jobs=envCons.coreProcess)(
            delayed(Model.__detectAndCorrectParallel)(
                i, columnName, columnData, pt, predictor, detector, modelPath
            )
            for i, (columnName, columnData) in enumerate(test.T.iteritems())
        )

        return corrected

    @staticmethod
    # @jit(parallel=True)
    def __detectAndCorrectParallel(
        i, columnName, columnData, pTransformer, predictor, detector, modelPath
    ):
        """ 
        Run the model on test data using the parallelization method for utilizing the machine power the 
        most and speed up the run time;\n
        Args:
            i (int):IDs of the train data - unique;
            columnName (str):       IDs of the train data - unique;
            columnData (df):        test data;
            pTransformer (object):  an instance of the Transformer class;
            predictor (object):     an instance of the AutoArima class;
            detector (object):      an instance of the Gaussian class;
            modelPath (str):        path to where the model is stored;
        Returns:
            corrected (df):         corrected data
        """
        try:
            st = datetime.now()
            log.trace("Iter = {} : Start = {}".format(i, columnName))

            modelFile = os.path.join(modelPath, columnName)

            ## Transform transformed test data
            testColumnData = (
                columnData.to_numpy().reshape((-1, 1)).astype(np.float64, copy=False)
            )
            l = testColumnData.shape[0]

            scalarFile, detectorModelFile, predictorModelFile = Model.__fileNames(
                modelFile
            )

            newTestId = False
            if (
                os.path.isfile(scalarFile)
                and os.path.isfile(detectorModelFile)
                and os.path.isfile(predictorModelFile)
            ):

                modelMap = map(
                    Model.__loadModels,
                    [pTransformer, detector, predictor],
                    [scalarFile, detectorModelFile, predictorModelFile],
                )

                pt, detect, predict = list(modelMap)

                testDataMean = pt.mean
                ## Replacing the NaN in test data with the mean of train data to avoid the detection of NaNs
                # and also to conform the transformation of test data as per the standard of train data
                testColumnData = np.where(
                    np.isnan(testColumnData), testDataMean, testColumnData
                )
                testColumnData, pt = Model.__transform(pt, testColumnData, train=False)
            else:
                log.warning(
                    "Data has not been trained ever before (new netz ID), so it's being trained now : {}".format(
                        columnName
                    )
                )
                testDataMean = np.nanmean(testColumnData)
                ## Load detector model if model is present else build it and save it;
                # If length of incoming data is greater than max window size, run validation on
                # incoming data or else repeat data to make it reach the size of maxWindow
                data = Model.__testDataForNewId(testColumnData, l)
                data = np.where(np.isnan(data), testDataMean, data)
                dl = len(data)

                pTransformer.mean = testDataMean
                data, pt = Model.__transform(pTransformer, data, train=True)
                detect, maxWindowSize = Model.__detector(detector, data, dl)

                ## Build the predictor
                predict = predictor.build(data[dl - maxWindowSize :])

                ## Save the trained models
                any(
                    map(
                        Model.__saveModels,
                        [pt, detect, predict],
                        [scalarFile, detectorModelFile, predictorModelFile],
                    )
                )

                newTestId = True

                ## Replacing the NaN in test data with the mean of train data to avoid the detection of NaNs
                # and also to conform the transformation of test data as per the standard of train data
                testColumnData = np.where(
                    np.isnan(testColumnData), testDataMean, testColumnData
                )
                ## Data to train new IDs have been prepared above, now data for testing need to be prepared
                testColumnData, pt = Model.__transform(pt, testColumnData, train=False)

            ## Get the epsilon
            epsilon = detect.eps
            std = detect.std

            log.trace(
                "Test data mean for {} = {}, std = {}".format(
                    columnName, testDataMean, std
                )
            )

            ## Run the detection and prediction on test data window, and after every window
            # update the model, update epsilon
            corrected = columnData.copy()

            for s in range(0, l, envCons.testDataSlice):
                ## Select the test data window
                testData = testColumnData[s : s + envCons.testDataSlice, :]

                ## Forecast no of steps = len(test data window) in future
                forecast, _ = predict.predict(steps=len(testData))
                forecast = forecast.reshape((-1, 1))

                ## Detect the pValue for test data
                pValue = detect.detect(testData)

                correctedTestData = Model.__correctedData(
                    detect, std, pValue, epsilon, testData, forecast
                )

                ## In case there are negative values in the corrected data, replace them with the
                # sum of the train data mean and the absolute value of the negative value
                correctedTransformed = Model.__inverseTransform(
                    pt, correctedTestData.copy()
                ).ravel()
                correctedTransformed = np.where(
                    correctedTransformed < 0,
                    testDataMean - correctedTransformed,
                    correctedTransformed,
                )
                corrected.iloc[s : s + envCons.testDataSlice] = correctedTransformed

                log.trace(
                    "columnName = {}, pValue : {} : epsilon = {} : std = {}".format(
                        columnName, pValue.ravel(), epsilon, std
                    )
                )

                log.trace(
                    "columnName = {}, testData = {} :============: {}, corrected = {} :============: {}".format(
                        columnName,
                        testData,
                        columnData.iloc[s : s + envCons.testDataSlice].values,
                        correctedTestData,
                        corrected.iloc[s : s + envCons.testDataSlice].values,
                    )
                )

                if not newTestId:
                    ## Update the model with new data
                    detect = detect.update(correctedTestData)
                    predict = predict.update(correctedTestData)

                    ## Update the new epsilon
                    pValueCorrected = detect.detect(correctedTestData)
                    epsilon = np.mean(np.concatenate(([[epsilon]], pValueCorrected)))
                    epsilon, std = Model.__epsilonAndStd(epsilon, std)

            ## Save updated model at the end of testing on window data
            if envCons.saveModelWhileTest and not newTestId:
                log.trace("Saving the updated model")

                ## Assign final epsilon to the object for validation next time
                finalCorrectedData = corrected.to_numpy().reshape((-1, 1))
                pt = pt.partial_fit(finalCorrectedData)

                ## Update the mean with corrected data
                correctedDataMean = pt.updateMean(finalCorrectedData, testDataMean, l)
                pt.mean = correctedDataMean

                detect.eps = epsilon
                any(
                    map(
                        Model.__saveModels,
                        [pt, detect, predict],
                        [scalarFile, detectorModelFile, predictorModelFile],
                    )
                )

            log.trace(
                "Iter = {} : End = {} : Elapsed time = {}".format(
                    i, columnName, (datetime.now() - st)
                )
            )
            return corrected
        except Exception as e:
            log.error(
                "Detection failed for columnName: {}, data {}, e: {}".format(
                    columnName, columnData, e
                )
            )

    @staticmethod
    @jit(parallel=True)
    def __fileNames(modelFile):
        """ Create file names for the model and scalar files. """
        scalarFile = modelFile + resCons.model.scalarFileExt
        detectorModelFile = (
            modelFile + resCons.model.detectorFile + resCons.model.modelFileExt
        )
        predictorModelFile = (
            modelFile + resCons.model.predictorFile + resCons.model.modelFileExt
        )
        return scalarFile, detectorModelFile, predictorModelFile

    @staticmethod
    @jit(parallel=True)
    def __correctedData(detect, std, pValue, epsilon, testData, forecast):
        """
        Prepare the final corrected data list and 
        select forecasted value if pValue is less than the threshold and the 
        std is less than or equals to the threshold else keep the test data;\n
        Args:
            detect (object):    detection model;
            std (float):        standard deviation;
            pValue (float):     p-value;       
            epsilon (float):    epsilon value;
            testData (series):  test data for correcting;
            forecast (series):  predicted values;
        Returns:
            correctedTestData (list): final list of corrected data;
        """
        mu = detect.mu
        sigma2 = detect.sigma2

        yL = mu - std * np.sqrt(sigma2)
        yH = mu + std * np.sqrt(sigma2)

        correctedTestData = np.where(
            (
                (~np.isnan(pValue))
                & (pValue < epsilon)
                & ((testData < yL) | (testData > yH))
            ),
            forecast,
            testData,
        )
        return correctedTestData

    @staticmethod
    @jit(parallel=True)
    def __loadModels(model, file):
        """ Load the model file."""
        return model.load(file)

    @staticmethod
    @jit(parallel=True)
    def __saveModels(model, file):
        """ Save the model file. """
        model.save(file)

    @staticmethod
    @jit
    def __transform(pt, data, train=True):
        """  
        Transform the train data with the Yeo-Johnson transformation;\n
        Args:
            pt (object):                    instance of the Transformer class;
            data (df):                      data to do the transformation on;        
            train (Bool):                   whether to train or not - determines if the pt fits and transforms or only transforms;
        Returns:
            transformedData[0] (series):    power-transformed data;
            pt (object):                    power transform with standardization included;
        """
        if train:
            transformedData = pt.fit_transform(data)
        else:
            transformedData = pt.transform(data)

        return transformedData, pt

    @staticmethod
    @jit
    def __inverseTransform(pt, data):
        """  
        Inverse transform the chosen transformer (standard scaler);\n
        Args:
            pt (object):    instance of the Transformer class;
            data (ndarray):      data to do the inverse transformation on;
        Returns:
            dataframe with reverse Yeo-Johnson transformation done;
        """
        return pt.inverse_transform(data)

    @staticmethod
    @jit(parallel=True)
    def __testDataForNewId(testData, l):
        """  
        Create train data from the test data in case a new id comes;\n
        Args:
            testData (ndarray): test data which doesn't have train data available, i.e. new id data;
            l (int):  length of the test data set;          
        Returns:
            data (ndarray): generated train data by filling test data with NaNs;
        """
        data = (
            np.concatenate(
                [testData[:, 0], np.repeat(np.nan, envCons.maxWindowSize - l),]
            ).reshape((-1, 1))
            if l < envCons.maxWindowSize
            else testData
        )

        return data

    @staticmethod
    @jit(parallel=True)
    def __epsilonAndStd(epsilon, std):
        """
        Define the threshold epsilon and standard deviation.
        """

        epsilon = (
            envCons.epsilon
            if (np.isnan(epsilon) | (epsilon > envCons.epsilon))
            else epsilon
        )

        std = (
            envCons.std
            if np.isnan(std) or np.isinf(std) or (std < envCons.std)
            else std
        )

        return epsilon, std