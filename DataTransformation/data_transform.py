from inspect import currentframe, getframeinfo
from os import path


class DataTransform:
    def __init__(self):
        self.FILE_BASENAME = path.basename(__file__)

    def replaceMissingWithNull(self, data, log_writer):
        """Replace missing with null

        Args:
            data(class 'pandas.core.frame.DataFrame'): Dataset
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'pandas.core.frame.DataFrame': Dataset
        """
        try:
            data.fillna("NULL", inplace=True)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "File Transformed successfully - Replace with NULL",
                "s",
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
