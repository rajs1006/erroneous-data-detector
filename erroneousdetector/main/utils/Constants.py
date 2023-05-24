import argparse
import multiprocessing
import os
from types import SimpleNamespace

import importlib_resources
import yaml
from dotenv import load_dotenv


def loadArguments():
    """  
    Method to load the arguments for execution.
    """
    parser = argparse.ArgumentParser(
        description="command line options to run Erroneous data detector app"
    )

    parser.add_argument("--env", help="pas full path environmnt file", required=True)
    parser.add_argument(
        "--action",
        choices=["gui", "cli"],
        help="whether to load gui or just to run the cli",
        default="cli",
    )

    parser.add_argument(
        "--timeseries",
        nargs="+",
        help="timeseries for which the app needs to run",
        default=None,
        required=False,
    )

    parser.add_argument(
        "--calibrate",
        help="run the data calibration",
        required=False,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--train",
        help="run the training",
        required=False,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--test",
        help="run the testing",
        required=False,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--plot",
        help="plot the unfiltered graph after train/test (useful for development purpose)",
        required=False,
        nargs="?",
        const=True,
        default=False,
    )
    args = parser.parse_args()

    if args.action == "gui" and (
        args.timeseries or args.calibrate or args.train or args.test or args.plot
    ):
        parser.error(
            "--timeseries --calibrate, --train, --test, --plot can only be set when --action=cli."
        )
    elif args.action == "cli" and (
        not args.calibrate and not args.train and not args.test
    ):
        parser.error(
            "at least one of the following options --calibrate, --train, --test  must be set"
        )

    return args


# Arguments Parser
# parse arguments, use this variable as a constant, to get the command line variable values
args = loadArguments()


def readEnvVar(envVar: str, defaultValue: any = None) -> str:
    """
    Get the value of an environment variable.\n
    Returns:
        the value of the environment variable;
    Throws:
        EnvironmentError if the environment variable isn't set;
    """
    val = os.getenv(envVar, defaultValue)

    if val is None:
        raise EnvironmentError("Environment variable '{}' must be set".format(envVar))
    return val


def checkFolderExist(folder):
    """  
    Check if the folder exists and if it doesn't, create it.
    """
    os.makedirs(folder, exist_ok=True)
    return folder


def checkPathExists(path, isHome=True):
    try:
        isPath = os.path.exists(path)
        if isPath:
            if isHome:
                os.chdir(path)
            else:
                pass
        else:
            raise Exception
    except:
        if isHome:
            raise Exception(
                "Location of HOME is not correct '{}', pass the path of the folder where your 'data' folder resides.".format(
                    path
                )
            )
        else:
            raise Exception(
                "Location of file is not correct '{}', make sure the file exists".format(
                    path
                )
            )
    return path


def loadResource(resourceFile: str):
    """
    This method loads the .yml resource file.\n
    Args:
        resourceFile : path of resource file;
    Returns:
        Dict: simplenamespace dict of resource variables;
    """
    with open(resourceFile) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class Environment:
    """  
    A class to encapsulate (and read) all of the constants stored in the environment file
    """

    # load the env file when the app is executed with Streamlit
    load_dotenv(args.env, verbose=True)

    home = checkPathExists(readEnvVar("homeFolderPath"), isHome=True)
    app = importlib_resources.files("erroneousdetector")

    # log-related variables
    logPath = checkFolderExist(readEnvVar("logPath"))
    logLevel = readEnvVar("logLevel")
    logFile = readEnvVar("logFile")

    # data and model paths
    saveResultPath = checkFolderExist(readEnvVar("saveResultPath"))
    historyFilePath = readEnvVar("historyFilePath")

    trainDataPath = checkFolderExist(readEnvVar("trainDataPath"))
    rawTrainDataPath = checkFolderExist(readEnvVar("rawTrainDataPath"))

    testDataPath = checkFolderExist(readEnvVar("testDataPath"))
    rawTestDataPath = checkFolderExist(readEnvVar("rawTestDataPath"))

    archiveResultPath = checkFolderExist(readEnvVar("archiveResultPath"))
    archiveTestPath = checkFolderExist(readEnvVar("archiveTestPath"))

    modelPath = checkFolderExist(readEnvVar("modelPath"))

    # strategies
    archiveStrategy = int(readEnvVar("archiveStrategy"))
    saveModelWhileTest = bool(int(readEnvVar("saveModelWhileTest")))

    # model parameters
    epsilon = float(readEnvVar("epsilon"))
    std = float(readEnvVar("std"))
    tsList = (readEnvVar("timeseriesList")).strip("][").split(",")

    # quality codes to get the data for training and testing
    trainQualityCode = readEnvVar("trainQualityCode")
    testQualityCode = readEnvVar("testQualityCode")

    # window selection parameters (for validation)
    startWindowSize = int(readEnvVar("startWindowSize"))
    maxWindowSize = int(readEnvVar("maxWindowSize"))
    windowStride = int(readEnvVar("windowStride"))

    # test data window selection size
    testDataSlice = int(readEnvVar("testDataSlice"))

    # system-related parameters for the multithreading process
    coreProcess = int(readEnvVar("coreProcess", multiprocessing.cpu_count()))
    scheduler = readEnvVar("scheduler", "threads")


class Resource:
    """
    A class to read the resource file.  
    """

    ## This file path is written here as this file is packaged with the build
    # and need not be accessed from outside the build (Build specific configurations)
    resourceFilePath = "resource/resource.yml"
    resource = checkPathExists(Environment.app / resourceFilePath, isHome=False)

    # load data
    dataResource = loadResource(resource)

    # properties for TRAIN and TEST data
    model = SimpleNamespace(**dataResource["MODEL"])
    data = SimpleNamespace(**dataResource["DATA"])
