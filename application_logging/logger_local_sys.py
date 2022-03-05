import logging
from datetime import datetime


class AppLogger:
    """This class shall be used for handling the logs."""

    def __init__(self, log_handler_name, file_name):

        # Define format for logs
        format = "%(asctime)s \t %(levelname)s \t %(message)s"
        current_timestamp = datetime.now()
        file_object = (
            file_name
            + "_"
            + str(current_timestamp.strftime("%d%m%Y"))
            + "_"
            + str(current_timestamp.strftime("%H%M%S"))
            + ".app"
        )

        # Create custom logger logging
        self.logger = logging.getLogger(log_handler_name)
        self.logger.setLevel(logging.INFO)

        self.file_handler = logging.FileHandler(file_object)

        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(logging.Formatter(format))
        self.logger.addHandler(self.file_handler)

    def log(self, file_basename, line_no, log_msg, log_mode="i"):
        """Log the data

        Args:
            file_basename (str): Base name of the source file
            line_no (int): Line number from where log message was raised
            log_msg (str): Log message
            log_mode (str, optional): E- Error, S- Success, I- Info.
                                        Defaults to "i".
        """
        basename_lineno = file_basename + "," + str(line_no)
        basename_lineno = basename_lineno.ljust(25)
        msg = basename_lineno + "\t\t" + log_msg

        if log_mode == "e":
            self.logger.error(msg)
        elif log_mode == "w":
            self.logger.warning(msg)
        elif log_mode == "i":
            self.logger.info(msg)
        elif log_mode == "c":
            self.logger.critical(msg)

    def stop_log(self, msg=""):
        """Stop the log

        Args:
            msg (str, optional): Log message. Defaults to "".
        """
        if len(msg) > 0:
            self.logger.error(msg)
        self.logger.info("\t__________ LOG STOPPED __________")
        self.file_handler.close()
        self.logger.removeHandler(self.file_handler)
        del self.file_handler
