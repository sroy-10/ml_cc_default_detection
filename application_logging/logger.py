from datetime import datetime

import pandas as pd
from common_utility import Utility


class AppLogger:
    """This class shall be used for handling the logs."""

    def __init__(self, file_name):

        # Define format for logs
        current_timestamp = datetime.now()
        self.log_file = (
            file_name
            + "_"
            + str(current_timestamp.strftime("%d%m%Y"))
            + "_"
            + str(current_timestamp.strftime("%H%M%S"))
            + ".csv"
        )

        # Create custom logger logging
        self.logger = pd.DataFrame(
            columns=["Timestamp", "Mode", "Program", "Line", "Message"]
        )

    def log(self, file_basename, line_no, log_msg, log_mode="i"):
        """Log the data

        Args:
            file_basename (str): Base name of the source file
            line_no (int): Line number from where log message was raised
            log_msg (str): Log message
            log_mode (str, optional): E- Error, S- Success, I- Info.
                                        Defaults to "i".
        """

        current_timestamp = datetime.now()
        mode = ""
        if log_mode == "e":
            mode = "error"
        elif log_mode == "w":
            mode = "warning"
        elif log_mode == "i":
            mode = "info"
        elif log_mode == "s":
            mode = "success"

        self.logger.loc[len(self.logger)] = [
            current_timestamp,
            mode,
            file_basename,
            str(line_no),
            log_msg,
        ]

    def stop_log(self, log_msg=""):
        """Stop the log

        Args:
            log_msg (str, optional): Log message. Defaults to "".
        """
        current_timestamp = datetime.now()
        if len(log_msg) > 0:
            self.logger.loc[len(self.logger)] = [
                current_timestamp,
                "",
                "",
                "",
                log_msg,
            ]
        # Log stopped
        self.logger.loc[len(self.logger)] = [
            current_timestamp,
            "",
            "",
            "",
            "******** LOG STOPPED ********",
        ]
        # Save file to S3 Bucket
        Utility().saveLogData(
            self.log_file,
            self.logger,
        )
