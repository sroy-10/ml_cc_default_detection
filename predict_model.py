from datetime import datetime
from inspect import currentframe, getframeinfo
from os import path

import pandas as pd

from application_logging.logger import AppLogger
from common_utility import Utility
from DataPreprocess.clustering import Clustering
from DataPreprocess.data_preprocess import DataPreprocess
from DataTransformation.data_transform import DataTransform
from DBOperations.dbOperations import DbOperations


class PredictModel:
    """This class is used for prediction"""

    def __init__(self):
        try:
            self.dbOperation = DbOperations()
            self.utility = Utility()
            self.dataTransform = DataTransform()
            self.preprocess = DataPreprocess()

            self.FILE_BASENAME = path.basename(__file__)
            self.RULES = self.utility.getJsonData()
            log_folder = self.RULES["LogFolder"]
            log_fname = path.join(
                log_folder["PredictionLog"],
                "predictModelLog",
            ).replace("\\", "/")
            self.log_writer = AppLogger(
                log_fname,  # File_name
            )

        except Exception as exception_msg:
            raise exception_msg

    def get_prediction_from_data(self):
        """Preprocess the data and get the prediction

        Raises:
            exception_msg: Captures the exception messages
        """
        try:
            DATA_FOLDER_RULES = self.RULES["DataFolder"]
            MODEL_FOLDER_RULES = self.RULES["ModelFolder"]
            self.log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Prediction of model started",
            )
            # Delete outputs from last run
            self.utility.deleteFile(
                DATA_FOLDER_RULES["PredictOutputFile"],
                "",  # filename
                ".csv",
                self.log_writer,
            )
            # Get the data
            data = self.utility.getFileData(
                filepath=DATA_FOLDER_RULES["PredictData"],
                filename="",
                log_writer=self.log_writer,
            )
            # ------ Preprocessing ------
            # Impute missing value
            data = self.preprocess.impute_missing_values(
                data, self.log_writer
            )
            # Scaling numerical columns
            data = self.preprocess.scale_numerical_columns(
                data, self.log_writer
            )
            # Encoding categorical columns
            data = self.preprocess.encode_categorical_columns(
                data, self.log_writer
            )
            # Load K-Means model
            model_kmeans = self.utility.loadModel(
                filepath=MODEL_FOLDER_RULES["KMeans"],
                filename="",
                log_writer=self.log_writer,
            )
            # Clustering
            X = data
            cluster = Clustering()
            # Predict & Determine the clusters
            X["clusters"] = model_kmeans.predict(X)
            no_of_clusters = X["clusters"].unique()
            final = pd.DataFrame()
            for i in no_of_clusters:
                cluster_data = X[X["clusters"] == i]
                cluster_data = cluster_data.drop(["clusters"], axis=1)
                model_name = cluster.find_correct_model_file(
                    MODEL_FOLDER_RULES["__root__"], i, self.log_writer
                )
                # Load Model
                model = self.utility.loadModel(
                    filepath=MODEL_FOLDER_RULES["__root__"],
                    filename=model_name,
                    log_writer=self.log_writer,
                )
                # Predict Model
                result = model.predict(cluster_data)
                # Append the result into prediction file
                tmp = pd.DataFrame(
                    {
                        # "ID": cluster_data.iloc[:, 0],
                        "Predictions": result,
                    }
                )

                # Concatination of temporary data with finals
                final = pd.concat([final, tmp], ignore_index=True)

            # Creation of CSV output file
            current_timestamp = datetime.now()
            predict_filename = (
                "Prediction"
                + "_"
                + str(current_timestamp.strftime("%d%m%Y"))
                + "_"
                + str(current_timestamp.strftime("%H%M%S"))
                + ".csv"
            )

            # predictionPath = path.join(
            #     DATA_FOLDER_RULES["PredictOutputFile"],
            #     predict_filename,
            # ).replace("\\", "/")
            # # Appends result to prediction file
            # final.to_csv(predictionPath, header=True, mode="a+")

            # Save prediction file to S3
            self.utility.saveFileData(
                DATA_FOLDER_RULES["PredictOutputFile"],
                predict_filename,
                final,
                self.log_writer,
            )

            # Log
            self.log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Prediction of model ended",
            )
        except Exception as exception_msg:
            raise exception_msg

        finally:
            # Stop log
            self.log_writer.stop_log()
