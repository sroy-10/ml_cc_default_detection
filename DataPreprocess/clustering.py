import os
from inspect import currentframe, getframeinfo
from os import path

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans


class Clustering:
    """This class is used to create clusters, find optimal,
    number of clusters and find the correct model of the clusters
    """

    def __init__(self):
        self.FILE_BASENAME = path.basename(__file__)

    def elbow_plot(self, data, log_writer):
        """Elbow plotting to determing optimum number of clusters

        Args:
            data (class 'pandas.core.frame.DataFrame'): Dataset
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            int: Number of optimal clusters
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Elbow plotting to determine optimum cluster started",
        )

        inertia = []
        try:
            for i in range(1, 11):
                # Initializing the KMeans object
                kmeans = KMeans(
                    n_clusters=i, init="k-means++", random_state=10
                )
                # Fitting the data to the KMeans Algorithm
                kmeans.fit(data)
                inertia.append(kmeans.inertia_)
            # Creating the graph between Inertia and the number of clusters
            plt.plot(range(1, 11), inertia)
            plt.title("The Elbow Method")
            plt.xlabel("Number of clusters")
            plt.ylabel("Inertia")

            # saving the elbow plot locally
            # plt.savefig(path.join(filepath, filename))

            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(
                range(1, 11),
                inertia,
                curve="convex",
                direction="decreasing",
            )
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Succes in determining optimum cluster. "
                + "The optimum number of clusters is: {cluster}".format(
                    cluster=str(kn.knee)
                ),
            )
            return kn.knee

        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error to determine optimum cluster. {e}".format(
                    e=exception_msg
                ),
                "e",
            )
            raise exception_msg

    def create_clusters(self, data, no_of_clusters, log_writer):
        """Create clusters

        Args:
            data (class 'pandas.core.frame.DataFrame'): Dataset
            no_of_clusters (int): Number of clusters to be created
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: [description]

        Returns:
           class 'pandas.core.frame.DataFrame': Dataset
           class 'sklearn.cluster._kmeans.KMeans': Kmeans Model

        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Creating cluster through KMeans started",
        )
        try:
            kmeans = KMeans(
                n_clusters=no_of_clusters,
                init="k-means++",
                random_state=10,
            )
            #  Divide data into clusters
            y_kmeans = kmeans.fit_predict(data)
            # Create a new column to identify the cluster information
            data["Cluster"] = y_kmeans
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Successfully created clusters",
            )
            return data, kmeans
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured while creating clusters. {e}".format(
                    e=exception_msg
                ),
            )
            raise exception_msg

    def find_correct_model_file(
        self, filepath, cluster_number, log_writer
    ):
        """Find the correct model file

        Args:
            filepath (str): Filepath of the model
            cluster_number (int): Cluster number
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Returns:
            str: Name of the optimal model file
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Finding of correct cluster started. Cluster Number: "
            + str(cluster_number),
        )

        for model_name in os.listdir(filepath):
            if path.isfile(path.join(filepath, model_name)):
                cn = int(model_name.split("_C_")[1].split(".")[0])
                if cn == cluster_number:
                    return model_name
