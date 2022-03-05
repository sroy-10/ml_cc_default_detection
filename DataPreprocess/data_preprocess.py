from inspect import currentframe, getframeinfo
from os import path

import numpy as np
import pandas as pd
from feature_engine.imputation import CategoricalImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocess:
    """This class will Preprocess the data"""

    def __init__(self):
        self.FILE_BASENAME = path.basename(__file__)

    def impute_missing_values(self, data, log_writer):
        """Imputing of missing values

        Args:
            data (class 'pandas.core.frame.DataFrame'): Dataset
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'pandas.core.frame.DataFrame': Dataset
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Imputing missing value started",
        )
        try:
            null_counts = data.isna().sum()
            null_col = []
            imputer = CategoricalImputer()
            for i in range(len(null_counts)):
                if null_counts[i] > 0:
                    col = null_counts.index[i]
                    null_col.append(col)
                    data[col] = imputer.fit_transform(data[col])
            # If there are any missing value column, list them in the log
            if len(null_col) != 0:
                log_writer.log(
                    self.FILE_BASENAME,
                    getframeinfo(currentframe()).lineno,
                    "Missing Value columns are: {col}".format(
                        col=null_col
                    ),
                )
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Imputing of missing value ended successfully",
            )
            return data
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured in imputing missing value. {e}".format(
                    e=exception_msg
                ),
                "e",
            )
            raise exception_msg

    def scale_numerical_columns(self, data, log_writer):
        """Scaling of the numerical columns

        Args:
            data (class 'pandas.core.frame.DataFrame'): Dataset
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'pandas.core.frame.DataFrame': Dataset
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Scaling for numerical column started",
        )
        # Extract numerical data
        try:
            numeric_dtype = [
                "int16",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
            ]
            num_df = data.select_dtypes(include=numeric_dtype)
            scaler = StandardScaler()
            for col in num_df.columns:
                data[col] = scaler.fit_transform(
                    np.asarray(data[col]).reshape(-1, 1)
                )
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Scaling for numerical column ended",
            )
            return data
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured in scaling numerical column. {e}.".format(
                    e=exception_msg
                ),
                "e",
            )
            raise exception_msg

    def encode_categorical_columns(self, data, log_writer):
        """Encoding of the categorical columns

        Args:
            data (class 'pandas.core.frame.DataFrame'): Dataset
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'pandas.core.frame.DataFrame': Dataset
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Encoding of categorical column started",
        )
        try:
            category_dtype = ["object"]
            cat_df = data.select_dtypes(include=category_dtype)
            for col in cat_df.columns:
                data[col] = pd.get_dummies(
                    data[col], prefix=col, drop_first=True
                )
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Encoding of categorical column ended",
            )
            return data
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured in encoding categorical column. {e}".format(
                    e=exception_msg
                ),
                "e",
            )
            raise exception_msg

    def handle_imbalanced_dataset(self, x, y, log_writer):
        """Handling of imbalanced dataset

        Args:
            x (class 'pandas.core.frame.DataFrame'): Independent features
            y (class 'pandas.core.frame.DataFrame'): Target column
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'pandas.core.frame.DataFrame': Independent features
            class 'pandas.core.frame.DataFrame': Target column
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Dataset balancing started.",
            # + " Initial number of records [0, 1]: "
            # + (", ").join(list(pd.Series(y).value_counts())),
        )
        try:
            sample = RandomOverSampler(random_state=42)
            x_sample, y_sample = sample.fit_resample(x, y)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Dataset balancing successful",
                # + " Final number of records [0, 1]: "
                # + (", ").join(list(pd.Series(y_sample).value_counts())),
            )
            return x_sample, y_sample
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured in balancing dataset. {e}".format(
                    e=exception_msg
                ),
                "e",
            )
            raise exception_msg

    def separate_xy(self, data, label, log_writer):
        """Separate target from Independent fetaures

        Args:
            data (class 'pandas.core.frame.DataFrame'): Dataset
            label (str): target label name
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'pandas.core.frame.DataFrame': Dataset
            class 'pandas.core.frame.DataFrame': Dataset
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Separation of dependent and independent feature started",
        )
        try:
            y = data[label]
            X = data.drop(label, axis=1)
            # Label encoding the target column
            y = LabelEncoder().fit_transform(y)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Separation of dependent and independent feature"
                + " ended successfully.",
            )
            return X, y
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured in separating dependent "
                + "and independent features. {e}".format(
                    e=exception_msg
                ),
                "e",
            )
            raise exception_msg
