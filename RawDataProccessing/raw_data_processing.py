from inspect import currentframe, getframeinfo
from os import path

from common_utility import Utility


class RawDataProcess:
    """This class is used to process Raw uplaoded data"""

    def __init__(self):
        self.FILE_BASENAME = path.basename(__file__)

    def allowed_file(self, filename, allowed_extensions):
        """Allowed file check

        Args:
            filename (str): Source filename
            allowed_extensions (list): List of allowed extensions

        Returns:
            bool: True - if the allowed extension exist else, False
        """
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in allowed_extensions
        )

    def RawUploadFile(
        self,
        app,
        filename,
        file_obj,
        allowed_extensions,
        upload_folder,
        log_writer,
        content_type="",
    ):
        """Save the RAW uploaded file

        Args:
            app (class 'Flask'): Flask application
            filename (str): Source filename
            file_obj (str): Source file
            allowed_extensions (list): List of allowed extensions
            upload_folder (str): Flask upload folder
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging
            content_type (str): request content type

        Raises:
            Exception: Filename not found
            Exception: File not found
            Exception: File extension not allowed
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Start of file upload",
        )
        # Checking if file & filename is blank
        if filename == "":
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Filename not found",
                "e",
            )
            raise Exception("Filename not found")
        if not file_obj:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "File not found",
                "e",
            )
            raise Exception("File not found")
        # If the filename exists in the ALLOWED_EXTENSIONS,
        # save the file in UPLOAD_FOLDER
        if self.allowed_file(filename, allowed_extensions):
            # Configuring the Flask upload folder
            # app.config["UPLOAD_FOLDER"] = upload_folder
            # file_obj.save(
            #     path.join(app.config["UPLOAD_FOLDER"], filename)
            # )

            # Save file to S3 Bucket
            Utility().saveFileData(
                upload_folder,
                filename,
                file_obj,
                log_writer,
                convert="",
                content_type=content_type,
            )

            # Writing to log
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "File uploaded successfully",
                "s",
            )
        else:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "File extension not allowed",
                "e",
            )
            raise Exception("File extension not allowed")

    def validateColumnLength(self, data, no_of_column, log_writer):
        """Validate the column length

        Args:
            data(class 'pandas.core.frame.DataFrame'): Dataset
            no_of_column (int): Number of columns
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            Exception: Captures the exception messages
        """
        if data.shape[1] != no_of_column:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "FAILED VALIDATION - No. of column",
                "e",
            )
            raise Exception("No of column do not match")
        else:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "No. of column matches",
            )
