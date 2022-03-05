import json
import os
import pickle
from inspect import currentframe, getframeinfo
from os import path

import boto3
import pandas as pd
from dotenv import load_dotenv


class Utility:
    """This class is used to perform various utility functions"""

    def __init__(self):
        self.FILE_BASENAME = path.basename(__file__)

        # Load environmental variables
        load_dotenv()

        # Load S3 related parameters
        self.S3_BUCKET_ROOT = os.getenv("S3_BUCKET_ROOT")
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
        self.REGION_NAME = os.getenv("REGION_NAME")
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

        # Get the reference of the S3 storage
        self.s3 = boto3.resource(
            service_name="s3",
            region_name=self.REGION_NAME,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
        )
        self.s3_bucket = self.s3.Bucket(self.S3_BUCKET_NAME)

        self.s3_client = boto3.client(
            "s3",
            region_name=self.REGION_NAME,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
        )

    def getJsonData(self):
        """Get the JSON data from 'general_config.json'

        Raises:
            Exception: Captures the exception messages

        Returns:
            dict: JSON Data
        """
        rules = dict()
        try:
            # 1st JSON Data
            with open("general_config.json", "r") as f:
                json_data = json.load(f)
                f.close()
            hierarchy_folder = [
                "DataFolder",
                "LogFolder",
                "ModelFolder",
            ]
            for folder in hierarchy_folder:
                for i in json_data[folder]:
                    if i != "__root__":
                        json_data[folder][i] = path.join(
                            json_data[folder]["__root__"],
                            json_data[folder][i],
                        ).replace("\\", "/")
            rules = json_data
            return rules
        except Exception as exception_msg:
            raise Exception(
                "Error in reading JSON. " + str(exception_msg)
            )

    def moveFilesFromDir(self, source, dest, filename, log_writer):
        """Move files from one directory to another

        Args:
            source (str): The file path of the source
            dest (str): The file path of the destination
            filename (str): The filename which needs to be moved
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages
        """
        src = path.join(source, filename).replace("\\", "/")
        dst = path.join(dest, filename).replace("\\", "/")
        try:
            # Copy files from one local folder to another
            # shutil.move(src, dst)

            # Copy files from one folder to another in S3 & then Delete it
            copy_source = {"Bucket": self.S3_BUCKET_NAME, "Key": src}
            self.s3.meta.client.copy(
                copy_source, self.S3_BUCKET_NAME, dst
            )
            self.s3.Object(self.S3_BUCKET_NAME, src).delete()

            # writing the log
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "File {fname} has been moved to {dest}".format(
                    fname=filename, dest=dest
                ),
            )
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "File {fname} has not been moved to {dest}. {e}".format(
                    fname=filename, dest=dest, e=exception_msg
                ),
                "e",
            )
            raise exception_msg

    def deleteFile(self, filepath, filename, extension, log_writer):
        """Delete File from Directory

        Args:
            filepath (str): Filepath of the file that needs to be deleted
            filename (str): Name of the file that needs to be deleted
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages
        """
        try:
            # Remove the file from local directory
            # os.remove(src)

            # Remove the file from S3 bucket
            # If filename is mentioned, delete the particular file
            # If the filename is not mentioned, then delete all the file
            if len(filename) > 0:
                src = path.join(filepath, filename).replace("\\", "/")
                self.s3.Object(self.S3_BUCKET_NAME, src).delete()
            else:
                # Remove the files from S3 Bucket
                for object_summary in self.s3_bucket.objects.filter(
                    Prefix=filepath
                ):
                    if object_summary.key.endswith(extension):
                        self.s3.Object(
                            self.S3_BUCKET_NAME, object_summary.key
                        ).delete()

        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "File {fname} has been deleted from path: {fpath}. {e}".format(
                    fname=filename, fpath=filepath, e=exception_msg
                ),
                "e",
            )
            raise exception_msg

    def getFileData(self, filepath, filename, log_writer):
        """Read the data from file

        Args:
            filepath (str): Filepath of the file that needs to be read
            filename (str): Filename of the file that needs to be read
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            Exception: Custom exception message incase no record is found
            exception_msg: Captures the exception messages

        Returns:
            class 'pandas.core.frame.DataFrame': Dataset
        """
        try:
            # If the filename is not provided, read all the files of the path
            data = pd.DataFrame()
            if len(filename) > 0:
                file = path.join(filepath, filename).replace("\\", "/")
                # data = pd.read_csv(file)

                # Read file from S3 Bucket
                data = pd.read_csv(
                    f"{self.S3_BUCKET_ROOT}{file}",
                    storage_options={
                        "key": self.AWS_ACCESS_KEY_ID,
                        "secret": self.AWS_SECRET_ACCESS_KEY,
                    },
                )
            else:
                # Get the list of files from S3 bucket
                for object_summary in self.s3_bucket.objects.filter(
                    Prefix=filepath
                ):
                    if object_summary.key.endswith(".csv"):
                        file = object_summary.key
                        df_tmp = pd.read_csv(
                            f"{self.S3_BUCKET_ROOT}{file}",
                            storage_options={
                                "key": self.AWS_ACCESS_KEY_ID,
                                "secret": self.AWS_SECRET_ACCESS_KEY,
                            },
                        )
                        # Concat the records of all the files
                        data = pd.concat(
                            [data, df_tmp], ignore_index=True
                        )
                # for f in os.listdir(filepath):
                #     file = path.join(filepath, f)
                #     df_tmp = pd.read_csv(file)
                #     data = pd.concat([data, df_tmp], ignore_index=True)

            if len(data) == 0:
                log_writer.log(
                    self.FILE_BASENAME,
                    getframeinfo(currentframe()).lineno,
                    "Data read properly from: " + file,
                )
                raise Exception("No records found")

            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Data read properly from: " + file,
            )
            return data
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                str(exception_msg),
                "e",
            )
            raise exception_msg

    def saveFileData(
        self,
        filepath,
        filename,
        data,
        log_writer,
        convert="X",
        content_type="",
    ):
        """Save data to file

        Args:
            filepath (str): Filepath where data needs to be saved
            filename (str): Filename of the saved data
            data (class 'pandas.core.frame.DataFrame'): Dataset
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging
            convert (str): whether data needs to be converted to CSV
            content_type (str): request content type

        Raises:
            exception_msg: Captures the exception messages
        """
        try:
            # data.to_csv(path.join(filepath, filename), index=False)

            # Save file to S3 bucket
            file = path.join(filepath, filename).replace("\\", "/")
            # If the convert is not set, save the file as it is
            if convert == "X":
                data.to_csv(
                    f"{self.S3_BUCKET_ROOT}{file}",
                    index=False,
                    storage_options={
                        "key": self.AWS_ACCESS_KEY_ID,
                        "secret": self.AWS_SECRET_ACCESS_KEY,
                    },
                )
            else:
                self.s3_client.put_object(
                    Body=data,
                    Bucket=self.S3_BUCKET_NAME,
                    Key=file,
                    ContentType=content_type,
                )

            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "File: {fname} saved to {fpath}".format(
                    fname=filename, fpath=filepath
                ),
            )
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                str(exception_msg),
                "e",
            )
            raise exception_msg

    def saveLogData(
        self,
        file,
        data,
    ):
        """Save data to file

        Args:
            file (str): File path + File name
            data (class 'pandas.core.frame.DataFrame'): Dataset

        Raises:
            exception_msg: Captures the exception messages
        """
        try:
            # Save file to S3 bucket
            data.to_csv(
                f"{self.S3_BUCKET_ROOT}{file}",
                index=False,
                storage_options={
                    "key": self.AWS_ACCESS_KEY_ID,
                    "secret": self.AWS_SECRET_ACCESS_KEY,
                },
            )

        except Exception as exception_msg:
            raise exception_msg

    # def delModel(self, filepath, extension, log_writer):
    #     """Delete model

    #     Args:
    #         filepath (str): Filepath of the model
    #         log_writer (class 'application_logging.logger.AppLogger'):
    #                         Object for logging

    #     Raises:
    #         exception_msg: Captures the exception messages
    #     """
    #     try:
    #         # Scan if there are any models in the folder
    #         # If found, delete the model
    #         log_writer.log(
    #             self.FILE_BASENAME,
    #             getframeinfo(currentframe()).lineno,
    #             "Deleting of model started",
    #         )

    #         # Remove the files from local directory
    #         # for fname in os.listdir(filepath):
    #         #     file = path.join(filepath, fname)
    #         #     if path.isfile(file):
    #         #         os.remove(file)
    #     try:
    #         # Remove the file from local directory
    #         # os.remove(src)

    #         # Remove the files from S3 Bucket
    #         for object_summary in self.s3_bucket.objects.filter(
    #             Prefix=filepath
    #         ):
    #             if object_summary.key.endswith(extension):
    #                 self.s3.Object(
    #                     self.S3_BUCKET_NAME, object_summary.key
    #                 ).delete()

    #         # Log
    #         log_writer.log(
    #             self.FILE_BASENAME,
    #             getframeinfo(currentframe()).lineno,
    #             "Deleting of model ended",
    #         )
    #     except Exception as exception_msg:
    #         log_writer.log(
    #             self.FILE_BASENAME,
    #             getframeinfo(currentframe()).lineno,
    #             "Error in deleting model. {e}".format(
    #                 e=str(exception_msg)
    #             ),
    #             "e",
    #         )
    #         raise exception_msg

    def saveModel(self, model, filepath, filename, log_writer):
        """Save model

        Args:
            model (class any): Model object that needs to be saved
            filepath (str): Filepath where the model needs to be saved
            filename (str): Filename of the model that needs to be saved
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Saving of the model started",
        )
        try:
            # Save the model to file provided
            # with open(path.join(filepath, filename), "wb") as f:
            #     pickle.dump(model, f)

            file = path.join(filepath, filename).replace("\\", "/")

            # Save the model in S3 bucket
            pickle_byte_obj = pickle.dumps(model)
            self.s3.Object(self.S3_BUCKET_NAME, file).put(
                Body=pickle_byte_obj
            )

            # Log the status
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Saving of the model ended. Saved in: "
                + str(path.join(filepath, filename).replace("\\", "/")),
            )
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error in saving model. {e}".format(
                    e=str(exception_msg)
                ),
                "e",
            )
            raise exception_msg

    def loadModel(
        self, filepath, filename, log_writer, extension=".sav"
    ):
        """Load model

        Args:
            filepath (str): Filepath of the model that needs to be loaded
            filename (str): Filename of the model that needs to be loaded
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging
            extension(str): Extension of the model

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class any: Model object
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Loading of the model started",
        )
        try:
            # # Filename not provided & multiple models are present
            # # In that case, get the latest model
            # if len(filename) == 0:
            #     f = glob(path.join(filepath, "*"))
            #     file = max(f, key=os.path.getctime)
            # else:
            #     file = path.join(filepath, filename)

            # # Load the model
            # with open(file, "rb") as f:
            #     log_writer.log(
            #         self.FILE_BASENAME,
            #         getframeinfo(currentframe()).lineno,
            #         "Model loadedd successfully",
            #     )
            #     return pickle.load(f)

            # Load the model from S3 bucket

            if len(filename) > 0:
                file = path.join(filepath, filename).replace("\\", "/")
                response = self.s3.Object(
                    self.S3_BUCKET_NAME, file
                ).get()
            else:
                for object_summary in self.s3_bucket.objects.filter(
                    Prefix=filepath
                ):
                    if object_summary.key.endswith(extension):
                        file = object_summary.key
                        response = self.s3.Object(
                            self.S3_BUCKET_NAME, file
                        ).get()

            return pickle.loads(response["Body"].read())

        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error while loading model. {e}".format(
                    e=str(exception_msg)
                ),
                "e",
            )
            raise exception_msg
