import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from erroneousdetector.main.build.Model import Model
from erroneousdetector.main.data.InputTest import RawTest
from erroneousdetector.main.data.InputTrain import RawTrain
from erroneousdetector.main.data.Test import Test
from erroneousdetector.main.data.Train import Train
from erroneousdetector.main.data.utils.Data import Data
from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)


# ignore Streamlit's deprecation warnings
st.set_option("deprecation.showfileUploaderEncoding", False)

if __name__ == "__main__":

    model = Model()
    rawTest = RawTest()
    test = Test()
    rawTrain = RawTrain()
    train = Train()

    st.title("Anomaly Detector")
    """ If you are running the app for the first time, you will need to calibrate it first. 
    """
    """
        To do so, first make sure that all of the data you wish to use for training is stored in **_data/train_** directory.
        Once the data is there, press the **CALIBRATE** button on the side and wait until the calibration is done.  
    """
    """  
    ### Instructions for running the detector:
    1. Upload your test data through the **Provide test data** button. To check if the test data has been successfully uploaded, press the same button again. 
    2. Train the model by pressing the **TRAIN** button. 
    3. Run the anomaly detection and correction model by pressing the **Detect And Correct** button. 

    Once the model has finished running, you can find the corrected CSV file in the **_data/results_** directory.
    """

    st.sidebar.header("Chose the operation mode:")

    # BUTTONS:
    calibrateButton = st.sidebar.button("CALIBRATE")
    testButton = st.sidebar.button("Provide test data")
    trainButton = st.sidebar.button("TRAIN")
    modelButton = st.sidebar.button("Detect And Correct")

    # CALIBRATION
    if calibrateButton:
        st.header("App Calibration")
        """ 
        This operation mode calibrates the app. It takes train data and transforms it to a format desired for training.
        """

        with st.spinner("Processing..."):
            start_calib = time.time()
            rawTrain.get()

            for ts in envCons.tsList:
                start = datetime.now()
                log.info("Calibration started... : {}".format(ts))

                # transforms the raw data into timeseries-specific dataframes
                rawTrain.transformAndSave(ts)

                # save calibration info into the History.txt file
                Data.saveHistory(ts, calib=True)

                log.info(
                    "Calibration ended : {} = {}".format(ts, datetime.now() - start)
                )

        # time printing related
        end_calib = time.time()
        hours_calib, rem_calib = divmod(end_calib - start_calib, 3600)
        minutes_calib, seconds_calib = divmod(rem_calib, 60)

        st.success(
            "Calibration is done. Your train data is now stored in **_data/train_** directory and is ready to be used. Processing time was: {:0>2}:{:0>2}:{:0>2}".format(
                int(hours_calib), int(minutes_calib), round(seconds_calib)
            )
        )

    # TEST DATA INPUT
    if testButton:
        st.header("Test Data")

        testFile = st.file_uploader("Upload the daily data csv file", type="csv")

        # LOAD THE TEST DATA
        if testFile:
            with st.spinner("Uploading..."):
                testDF = pd.DataFrame(pd.read_csv(testFile, parse_dates=["GASTAG"]))
                rawTest.getStreamlit(testDF)

                for ts in envCons.tsList:
                    rawTest.transformAndSave(ts)

            st.success("Your test data has been successfully uploaded.")

    # TRAINING
    if trainButton:
        st.header("Train the model")
        """ 
        This operation mode trains the anomaly detection and correction model with the available train data.
        """

        start_train = time.time()

        for timeseries in envCons.tsList:
            start_ts = time.time()

            with st.spinner("Training for {} timeseries...".format(timeseries)):
                start = datetime.now()
                log.info("Training started... : {}".format(ts))

                # get the train data
                train = Train()
                trainData = train.get(ts)

                # train and validate the detection model
                m = model(timeseries)
                m.trainAndValidate(trainData)

                # save training info into the History.txt file
                Data.saveHistory(ts, train=trainData)

                log.info("Training ended : {} = {}".format(ts, datetime.now() - start))

            # time printing related - ts
            end_ts = time.time()
            hours_ts, rem_ts = divmod(end_ts - start_ts, 3600)
            minutes_ts, seconds_ts = divmod(rem_ts, 60)

            st.success(
                "The model for {} timeseries has been **successfully** trained. Training time was: {:0>2}:{:0>2}:{:0>2}".format(
                    timeseries, int(hours_ts), int(minutes_ts), round(seconds_ts)
                )
            )

        # time printing related - whole train part
        end_train = time.time()
        hours_train, rem_train = divmod(end_train - start_train, 3600)
        minutes_train, seconds_train = divmod(rem_train, 60)

        st.success(
            "The model has been trained for **all timeseries**. You can now run the model to detect and correct anomalies. Total training time was: {:0>2}:{:0>2}:{:0>2}.".format(
                int(hours_train), int(minutes_train), round(seconds_train)
            )
        )

    # TESTING
    if modelButton:
        st.header("Find and Rectify Anomalies")
        """ 
        This operation mode runs the anomaly detection and correction. 
        Once it finishes running the model it saves the corrected .csv file in the _**data/results**_ directory.
        """
        start_model = time.time()

        for timeseries in envCons.tsList:
            start = datetime.now()
            start_ts = time.time()
            log.info("Detection started... : {}".format(ts))

            with st.spinner(
                "Running the model for {} timeseries...".format(timeseries)
            ):

                testData = test.get(timeseries)
                rawTestData = test.get(timeseries, transform=False)

            with st.spinner("Running the model..."):
                m = model(timeseries)

                # detect and corrected anomalies in the test data
                correctedData = m.detectAndCorrect(testData)

                # append the corrected value to the test data csv file and transform it
                originalAndCorrected = test.mergeCorrectedAndRawTestData(
                    test.get(ts, transform=False), correctedData
                )

                # save the result
                Data.saveResult(originalAndCorrected)

                # save the testing info into History.txt file
                Data.saveHistory(ts, test=testData)

                log.info("Detection ended : {} = {}".format(ts, datetime.now() - start))

                # time printing related - ts
                end_ts = time.time()
                hours_ts, rem_ts = divmod(end_ts - start_ts, 3600)
                minutes_ts, seconds_ts = divmod(rem_ts, 60)

                st.success(
                    "{} timeseries has been **successfully** processed. Processing time was: {:0>2}:{:0>2}:{:0>2}.".format(
                        timeseries, int(hours_ts), int(minutes_ts), round(seconds_ts)
                    )
                )

        # time printing related
        end_model = time.time()
        hours_model, rem_model = divmod(end_model - start_model, 3600)
        minutes_model, seconds_model = divmod(rem_model, 60)

        st.success(
            "Your data has been corrected. You can find it in the **_data/result_**. Total processing time was: {:0>2}:{:0>2}:{:0>2}.".format(
                int(hours_model), int(minutes_model), round(seconds_model)
            )
        )

