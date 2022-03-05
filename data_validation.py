from datetime import datetime
from os import path

from werkzeug.utils import secure_filename

from application_logging.logger import AppLogger
from common_utility import Utility
from DataTransformation.data_transform import DataTransform
from DBOperations.dbOperations import DbOperations
from RawDataProccessing.raw_data_processing import RawDataProcess


class DataValidation:
    """Class to validate the incoming data"""

    def __init__(self, mode):
        """Constructor method

        Args:
            mode (str): Determines training or prediction

        Raises:
            exception_msg: Captures the exception messages
        """
        try:
            self.rawdata = RawDataProcess()
            self.dbOperation = DbOperations()
            self.utility = Utility()
            self.dataTransform = DataTransform()

            self.FILE_BASENAME = path.basename(__file__)
            self.RULES = self.utility.getJsonData()
            log_folder = self.RULES["LogFolder"]

            if mode == "training":
                log_fname = path.join(
                    log_folder["TrainValidationLog"],
                    "TrainValidationLog",
                ).replace("\\", "/")
            else:
                log_fname = path.join(
                    log_folder["PredictionValidationLog"],
                    "PredictionValidationLog",
                ).replace("\\", "/")
                # Remove the target column from JSON
                t = self.RULES["TargetColumnName"]
                del self.RULES["ColDetail"][t]
                # Reduce the number of columns
                self.RULES["NumberofColumns"] = (
                    self.RULES["NumberofColumns"] - 1
                )

            # Applogger
            self.log_writer = AppLogger(
                log_fname,
            )

        except Exception as exception_msg:
            raise exception_msg

    def validate_train_data(self, app, file_obj, mode, content_type):
        """Validate train data

        Args:
            app (class 'flask.app.Flask'): Flask app
            file_obj (str): Uploaded file object
            mode (str): Determines whether to delete DB table or not
            content_type (str): request content type

        Raises:
            exception_msg: Captures the exception messages
        """
        try:
            TABLE_NAME = self.RULES["DBTrainingTable"]
            FILENAME = secure_filename(file_obj.filename)
            DATA_FOLDER_RULES = self.RULES["DataFolder"]
            msg = ""
            # The variable determines where the file is present currently
            current_file_loc = DATA_FOLDER_RULES[
                "RawTrainDataUploadFolder"
            ]
            # Archiving raw upload file into the directory
            self.rawdata.RawUploadFile(
                app,
                FILENAME,
                file_obj,
                self.RULES["AllowedExtension"],
                DATA_FOLDER_RULES["RawTrainDataUploadFolder"],
                self.log_writer,
                content_type,
            )

            # Get the data from the current file location
            data = self.utility.getFileData(
                current_file_loc, FILENAME, self.log_writer
            )

            # Validate raw data - No. of columns, datatype, etc.
            self.rawdata.validateColumnLength(
                data,
                self.RULES["NumberofColumns"],
                self.log_writer,
            )

            # Data pre-processing - check for NULL value
            data = self.dataTransform.replaceMissingWithNull(
                data,
                self.log_writer,
            )

            # Save the data into the Good folder
            self.utility.saveFileData(
                DATA_FOLDER_RULES["GoodTrainDataFolder"],
                FILENAME,
                data,
                self.log_writer,
            )

            # Delete the file from RAW folder
            self.utility.deleteFile(
                DATA_FOLDER_RULES["RawTrainDataUploadFolder"],
                FILENAME,
                ".csv",
                self.log_writer,
            )

            current_file_loc = DATA_FOLDER_RULES["GoodTrainDataFolder"]

            # Create DB Table if not created
            if mode == "del_model":
                # Delete existing table
                self.dbOperation.deleteTable(
                    TABLE_NAME, self.log_writer
                )
            # Check if DB table exist or not
            if not self.dbOperation.checkTableExist(
                TABLE_NAME, self.log_writer
            ):
                # Create existing table
                self.dbOperation.createTable(
                    TABLE_NAME, self.RULES["ColDetail"], self.log_writer
                )
            # Get the file Data
            data = self.utility.getFileData(
                DATA_FOLDER_RULES["GoodTrainDataFolder"],
                FILENAME,
                self.log_writer,
            )
            # Save data to DB
            no_rec_insert, error_rec = self.dbOperation.insertTableRec(
                TABLE_NAME,
                self.RULES["ColDetail"],
                data,
                self.log_writer,
            )
            # Incase, no record was inserted in DB, Stop the process
            if no_rec_insert == 1:
                raise Exception("No record inserted in DB")

            # Incase of any error, while saving in DB
            # Store error records separately in a file
            if len(error_rec) != 0:
                current_timestamp = datetime.now()
                error_filename = (
                    "Error Records"
                    + "_"
                    + str(current_timestamp.strftime("%d%m%Y"))
                    + "_"
                    + str(current_timestamp.strftime("%H%M%S"))
                    + ".csv"
                )
                # Save the error records
                self.utility.saveFileData(
                    DATA_FOLDER_RULES["ErrorDBInsertData"],
                    error_filename,
                    error_rec,
                    self.log_writer,
                )
            # Delete the good training data from directory
            self.utility.deleteFile(
                DATA_FOLDER_RULES["GoodTrainDataFolder"],
                FILENAME,
                ".csv",
                self.log_writer,
            )
        except Exception as exception_msg:
            msg = exception_msg
            # Incase of any error, move file to bad folder
            self.utility.moveFilesFromDir(
                current_file_loc,  # source
                DATA_FOLDER_RULES["BadTrainDataFolder"],  # dest
                FILENAME,
                self.log_writer,
            )
            raise exception_msg
        finally:
            # Stop log
            if len(msg) == 0:
                self.log_writer.stop_log()
            else:
                self.log_writer.stop_log(str(msg))

    def validate_predict_data(self, app, file_obj, content_type):
        """Validate predict data

        Args:
            app (class 'flask.app.Flask'): Flask app
            file_obj (str): Uploaded file object
            content_type (str): request content type

        Raises:
            exception_msg: Captures the exception messages
        """
        try:
            FILENAME = secure_filename(file_obj.filename)
            DATA_FOLDER_RULES = self.RULES["DataFolder"]
            current_file_loc = DATA_FOLDER_RULES[
                "RawPredictDataUploadFolder"
            ]
            msg = ""
            # Archiving raw upload file into the directory
            self.rawdata.RawUploadFile(
                app,
                FILENAME,
                file_obj,
                self.RULES["AllowedExtension"],
                DATA_FOLDER_RULES["RawPredictDataUploadFolder"],
                self.log_writer,
                content_type,
            )

            # Get the data from the current file location
            data = self.utility.getFileData(
                current_file_loc, FILENAME, self.log_writer
            )

            # Validate raw data - No. of columns, datatype, etc.
            self.rawdata.validateColumnLength(
                data,
                self.RULES["NumberofColumns"],
                self.log_writer,
            )

            # Data pre-processing - check for NULL value
            data = self.dataTransform.replaceMissingWithNull(
                data,
                self.log_writer,
            )

            # Save the data into the current folder
            self.utility.saveFileData(
                DATA_FOLDER_RULES["GoodPredictDataFolder"],
                FILENAME,
                data,
                self.log_writer,
            )

            # Delete the file from RAW folder
            self.utility.deleteFile(
                DATA_FOLDER_RULES["RawPredictDataUploadFolder"],
                FILENAME,
                ".csv",
                self.log_writer,
            )

            current_file_loc = DATA_FOLDER_RULES[
                "GoodPredictDataFolder"
            ]

            # Move Data from 'good' to 'predict' folder
            self.utility.moveFilesFromDir(
                DATA_FOLDER_RULES["GoodPredictDataFolder"],  # source
                DATA_FOLDER_RULES["PredictData"],  # dest
                FILENAME,
                self.log_writer,
            )
        except Exception as exception_msg:
            msg = exception_msg
            # Incase of any error, move file to bad folder
            self.utility.moveFilesFromDir(
                current_file_loc,  # source
                DATA_FOLDER_RULES["BadPredictDataFolder"],  # dest
                FILENAME,
                self.log_writer,
            )
            raise exception_msg
        finally:
            # Stop log
            if len(msg) == 0:
                self.log_writer.stop_log()
            else:
                self.log_writer.stop_log(str(msg))
