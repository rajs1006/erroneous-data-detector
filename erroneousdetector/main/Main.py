import time
from datetime import datetime

import pandas as pd


from erroneousdetector.main.build.Model import Model
from erroneousdetector.main.data.InputTest import RawTest
from erroneousdetector.main.data.InputTrain import RawTrain
from erroneousdetector.main.data.Test import Test
from erroneousdetector.main.data.Train import Train
from erroneousdetector.main.data.utils.Data import Data
from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.Constants import args
from erroneousdetector.main.utils.logger.System import logger

log = logger(__file__)


def main():
    """  
    Main method to run the app in the cli mode.
    """

    # write into History.txt that the app was initiated
    Data.saveHistory(start=True)

    # initiate the model
    model = Model()

    # move the old result file to the archive folder
    Data.moveToArchive(result=True)

    # an individual timeseries can be passed either from the command line
    # or through the environment file as a list
    tsList = args.timeseries if args.timeseries is not None else envCons.tsList
    for ts in tsList:
        m = model(ts)

        trainData = None
        testData = None
        correctedData = None

        # CALIBRATION
        if args.calibrate:
            start = datetime.now()
            log.info("Calibration started... : {}".format(ts))

            # load the raw train data
            rawTrain = RawTrain()
            rawTrain.get()

            # transforms the raw train data into timeseries-specific dataframes
            rawTrain.transformAndSave(ts)

            # save calibration info into the History.txt file
            Data.saveHistory(ts, calib=True)

            log.info("Calibration ended : {} = {}".format(ts, datetime.now() - start))

        # TRAINING
        if args.train:
            start = datetime.now()
            log.info("Training started... : {}".format(ts))

            # get the train data
            train = Train()
            trainData = train.get(ts)

            # train and validate the detection model
            m.trainAndValidate(trainData)

            # save training info into the History.txt file
            Data.saveHistory(ts, train=trainData)

            log.info("Training ended : {} = {}".format(ts, datetime.now() - start))

        # TESTING
        if args.test:
            start = datetime.now()
            log.info("Detection started... : {}".format(ts))

            # load the test data and transform it by timeseries
            rawTest = RawTest()
            rawTest.get()

            # transforms the raw test data into timeseries-specific dataframes
            rawTest.transformAndSave(ts)

            # get the test data
            test = Test()
            testData = test.get(ts)

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

        # PLOTTING
        if args.plot:
            m.plot(ts, trainData, testData, correctedData)

    # move the used test file to the archive folder
    Data.moveToArchive(test=True)


if __name__ == "__main__":
    main()
