from inspect import currentframe, getframeinfo
from os import path

from sklearn.model_selection import train_test_split

from application_logging.logger import AppLogger
from common_utility import Utility
from DataPreprocess.clustering import Clustering
from DataPreprocess.data_preprocess import DataPreprocess
from DataTransformation.data_transform import DataTransform
from DBOperations.dbOperations import DbOperations
from ModelTuner.model_tuner import ModelTuner


class TrainModel:
    """This class is used to train the model"""

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
                log_folder["TrainningLog"],
                "trainModelLog",
            ).replace("\\", "/")
            self.log_writer = AppLogger(
                log_fname,  # File_name
            )

        except Exception as exception_msg:
            raise exception_msg

    def train_model_from_data(self):
        """Preprocess the data and train the model from data

        Raises:
            exception_msg: Captures the exception messages
        """
        try:
            TABLE_NAME = "training_data"
            DATA_FOLDER_RULES = self.RULES["DataFolder"]
            MODEL_FOLDER_RULES = self.RULES["ModelFolder"]
            self.log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Training of model started",
            )
            # Get the training data from DB & save it to 'TRAIN' folder
            FILENAME, data = self.dbOperation.getDbDataToCsv(
                TABLE_NAME,
                self.RULES["ColDetail"],
                self.log_writer,
            )
            # Save the data into S3 bucket
            self.utility.saveFileData(
                DATA_FOLDER_RULES["TrainData"],
                FILENAME,
                data,
                self.log_writer,
            )
            # Read the data from folder
            data = self.utility.getFileData(
                DATA_FOLDER_RULES["TrainData"],
                FILENAME,
                self.log_writer,
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
            # Separating target and independent features
            X, y = self.preprocess.separate_xy(
                data, self.RULES["TargetColumnName"], self.log_writer
            )
            # Handling imbalance dataset
            X, y = self.preprocess.handle_imbalanced_dataset(
                X, y, self.log_writer
            )
            # Delete existing models from last run
            for p in MODEL_FOLDER_RULES:
                self.utility.deleteFile(
                    MODEL_FOLDER_RULES[p], "", ".sav", self.log_writer
                )

            # ------ Clustering ------
            # Train model by reading the data
            cluster = Clustering()
            no_of_clusters = cluster.elbow_plot(
                X,
                self.log_writer,
            )
            # Create cluster
            X, kmeans_model = cluster.create_clusters(
                X, no_of_clusters, self.log_writer
            )
            # The below column is added to get the corresponding value of
            # the label while grouping the cluster
            X["Labels"] = y

            # Save model
            self.utility.saveModel(
                kmeans_model,
                MODEL_FOLDER_RULES["KMeans"],
                "KMeans.sav",
                self.log_writer,
            )
            list_of_clusters = X["Cluster"].unique()
            for i in list_of_clusters:
                # Filter the data of a particular cluster
                cluster_data = X[X["Cluster"] == i]
                cluster_x = cluster_data.drop(
                    ["Labels", "Cluster"], axis=1
                )
                cluster_y = cluster_data["Labels"]
                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_x,
                    cluster_y,
                    test_size=1 / 3,
                    random_state=49,
                )
                # Scaling numerical columns
                x_train = self.preprocess.scale_numerical_columns(
                    x_train, self.log_writer
                )
                # Getting the best model for each of the clusters
                model_finder = ModelTuner()
                (
                    best_model_name,
                    best_model_score,
                    best_model,
                ) = model_finder.get_best_model(
                    x_train, y_train, x_test, y_test, self.log_writer
                )
                # Saving the best model to the directory
                self.utility.saveModel(
                    best_model,
                    MODEL_FOLDER_RULES["__root__"],
                    best_model_name + "_C_" + str(i) + ".sav",
                    self.log_writer,
                )

        except Exception as exception_msg:
            self.log_writer.stop_log(str(exception_msg))
            raise exception_msg

        finally:
            # Stop log
            self.log_writer.stop_log()
