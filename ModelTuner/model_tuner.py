from inspect import currentframe, getframeinfo
from os import path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class ModelTuner:
    """This class is to tune the model and calculate the best model"""

    def __init__(self):
        self.FILE_BASENAME = path.basename(__file__)

    def get_best_params_for_dtree(self, train_x, train_y, log_writer):
        """Get the best parameter for Decision Tree

        Args:
            train_x (class 'pandas.core.frame.DataFrame'): Independent feature
            train_y (class 'pandas.core.frame.DataFrame'): Target feature
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'sklearn.tree._classes.DecisionTreeClassifier': DT model
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Getting the best params for Decision Tree",
        )
        try:
            # Initializing with different combination of parameters
            param_grid_dtree = {
                "criterion": ["gini", "entropy"],
                "max_depth": list(range(2, 4, 1)),
                "min_samples_leaf": list(range(5, 7, 1)),
                "random_state": [0, 50, 100],
            }
            # Creating an object of the Grid Search class
            cv = StratifiedKFold(
                n_splits=10, shuffle=True, random_state=1
            )
            grid = RandomizedSearchCV(
                DecisionTreeClassifier(),
                param_grid_dtree,
                verbose=3,
                cv=cv,
                n_jobs=-1,
            )
            # Finding the best parameters
            grid.fit(train_x, train_y)

            # Creating a new model with the best parameters
            dtree = DecisionTreeClassifier(
                criterion=grid.best_params_["criterion"],
                max_depth=grid.best_params_["max_depth"],
                min_samples_leaf=grid.best_params_["min_samples_leaf"],
                random_state=grid.best_params_["random_state"],
            )
            # Training the mew model
            dtree.fit(train_x, train_y)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Decision Tree model created."
                + " Decision Tree Parameter tuning completed."
                + " Best parameters for Decision Tree: "
                + str(grid.best_params_),
            )
            return dtree
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured while tuning Decision Tree model."
                + str(exception_msg),
                "e",
            )
            raise exception_msg

    def get_best_params_for_knn(self, train_x, train_y, log_writer):
        """Get the best parameter for KNN

        Args:
            train_x (class 'pandas.core.frame.DataFrame'): Independent feature
            train_y (class 'pandas.core.frame.DataFrame'): Target feature
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'sklearn.neighbors._classification.KNeighborsClassifier':
                    KNN model
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Getting the best params for KNN",
        )
        try:
            # Initializing with different combination of parameters
            param_grid_knn = {
                "n_neighbors": list(range(2, 5, 1)),
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            }
            # Creating an object of the Grid Search class
            cv = StratifiedKFold(
                n_splits=10, shuffle=True, random_state=1
            )
            grid = RandomizedSearchCV(
                KNC(),
                param_grid_knn,
                verbose=3,
                cv=cv,
                n_jobs=-1,
            )
            # Finding the best parameters
            grid.fit(train_x, train_y)

            # Creating a new model with the best parameters
            knc = KNC(
                n_neighbors=grid.best_params_["n_neighbors"],
                algorithm=grid.best_params_["algorithm"],
                n_jobs=-1,
            )
            # Training the mew model
            knc.fit(train_x, train_y)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "KNN model created."
                + " KNN Parameter tuning completed."
                + " Best parameters for KNN: "
                + str(grid.best_params_),
            )
            return knc
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured while tuning KNN model."
                + str(exception_msg),
                "e",
            )
            raise exception_msg

    def get_best_params_for_logreg(self, train_x, train_y, log_writer):
        """Get the best parameter for Logistic Regresion

        Args:
            train_x (class 'pandas.core.frame.DataFrame'): Independent feature
            train_y (class 'pandas.core.frame.DataFrame'): Target feature
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'sklearn.linear_model._logistic.LogisticRegression':
                    LogReg model
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Getting the best params for Logistic Regression",
        )
        try:
            # Initializing with different combination of parameters
            param_grid_logreg = {
                "penalty": ["l1", "l2"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "random_state": [0, 50, 100],
            }
            # Creating an object of the Grid Search class
            cv = StratifiedKFold(
                n_splits=10, shuffle=True, random_state=1
            )
            grid = RandomizedSearchCV(
                LogisticRegression(),
                param_grid_logreg,
                verbose=3,
                cv=cv,
                n_jobs=-1,
            )
            # Finding the best parameters
            grid.fit(train_x, train_y)

            # Creating a new model with the best parameters
            logreg = LogisticRegression(
                penalty=grid.best_params_["penalty"],
                C=grid.best_params_["C"],
                random_state=grid.best_params_["random_state"],
                n_jobs=-1,
            )
            # Training the mew model
            logreg.fit(train_x, train_y)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Logistic Regression model created."
                + " Logistic Regression Parameter tuning completed."
                + " Best parameters for Logistic Regression: "
                + str(grid.best_params_),
            )
            return logreg
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured while tuning Logistic Regression model."
                + str(exception_msg),
                "e",
            )
            raise exception_msg

    def get_best_params_for_rforest(self, train_x, train_y, log_writer):
        """Get the best parameters for random forest

        Args:
            train_x (class 'pandas.core.frame.DataFrame'): Independent feature
            train_y (class 'pandas.core.frame.DataFrame'): Target feature
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'sklearn.ensemble._forest.RandomForestClassifier':
                    Random Forest model
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Getting the best params for Random Forest",
        )
        try:
            # Initializing with different combination of parameters
            param_grid_rf = {
                "n_estimators": [50, 100, 130],
                "max_depth": range(3, 11, 1),
                "random_state": [0, 50, 100],
                "max_features": ["auto", "sqrt"],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            }
            # Creating an object of the Grid Search class
            cv = StratifiedKFold(
                n_splits=10, shuffle=True, random_state=1
            )
            grid = RandomizedSearchCV(
                RandomForestClassifier(),
                param_grid_rf,
                verbose=3,
                cv=cv,
                n_jobs=-1,
            )
            # Finding the best parameters
            grid.fit(train_x, train_y)

            # Creating a new model with the best parameters
            rf = RandomForestClassifier(
                n_estimators=grid.best_params_["n_estimators"],
                max_depth=grid.best_params_["max_depth"],
                random_state=grid.best_params_["random_state"],
                max_features=grid.best_params_["max_features"],
                min_samples_split=grid.best_params_[
                    "min_samples_split"
                ],
                min_samples_leaf=grid.best_params_["min_samples_leaf"],
                bootstrap=grid.best_params_["bootstrap"],
                n_jobs=-1,
            )
            # Training the mew model
            rf.fit(train_x, train_y)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Random Forest model created."
                + " Random Forest Parameter tuning completed."
                + " Best parameters for Random Forest: "
                + str(grid.best_params_),
            )
            return rf
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured while tuning Random Forest model."
                + str(exception_msg),
                "e",
            )
            raise exception_msg

    def get_best_params_for_svc(self, train_x, train_y, log_writer):
        """Get the best parameters for SVC

        Args:
            train_x (class 'pandas.core.frame.DataFrame'): Independent feature
            train_y (class 'pandas.core.frame.DataFrame'): Target feature
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'sklearn.svm._classes.SVC':
                    SVC model
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Getting the best params for SVC",
        )
        try:
            # Initializing with different combination of parameters
            param_grid_svc = {
                "C": [0.5, 0.7, 0.9, 1],
                "kernel": ["rbf", "poly", "sigmoid", "linear"],
                "random_state": [0, 50, 100],
            }
            # Creating an object of the Grid Search class
            cv = StratifiedKFold(
                n_splits=10, shuffle=True, random_state=1
            )
            grid = RandomizedSearchCV(
                SVC(),
                param_grid_svc,
                verbose=3,
                cv=cv,
                n_jobs=-1,
            )
            # finding the best parameters
            grid.fit(train_x, train_y)

            # creating a new model with the best parameters
            svc = SVC(
                C=grid.best_params_["C"],
                kernel=grid.best_params_["kernel"],
                random_state=grid.best_params_["random_state"],
            )
            # training the mew model
            svc.fit(train_x, train_y)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "SVC model created."
                + " SVC Parameter tuning completed."
                + " Best parameters for SVC: "
                + str(grid.best_params_),
            )
            return svc
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured while tuning SVC model."
                + str(exception_msg),
                "e",
            )
            raise exception_msg

    def get_best_params_for_xgboost(self, train_x, train_y, log_writer):
        """Get the best parameters for XGBoost model

        Args:
            train_x (class 'pandas.core.frame.DataFrame'): Independent feature
            train_y (class 'pandas.core.frame.DataFrame'): Target feature
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'xgboost.sklearn.XGBClassifier':
                    XGBoost model
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Getting the best params for XG Boost",
        )
        try:
            # initializing with different combination of parameters
            param_grid_xgboost = {
                "n_estimators": [50, 100, 130],
                "max_depth": range(3, 11, 1),
                "random_state": [0, 50, 100],
            }
            # Creating an object of the Ramdomized Search class
            cv = StratifiedKFold(
                n_splits=10, shuffle=True, random_state=1
            )
            grid = RandomizedSearchCV(
                XGBClassifier(objective="binary:logistic"),
                param_grid_xgboost,
                verbose=3,
                cv=cv,
                n_jobs=-1,
            )

            # finding the best parameters
            grid.fit(train_x, train_y)

            # creating a new model with the best parameters
            xgb = XGBClassifier(
                random_state=grid.best_params_["random_state"],
                max_depth=grid.best_params_["max_depth"],
                n_estimators=grid.best_params_["n_estimators"],
            )
            # training the mew model
            xgb.fit(train_x, train_y)
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "XGBoost model created."
                + " XGBoost Parameter tuning completed."
                + " Best parameters for XGBoost: "
                + str(grid.best_params_),
            )
            return xgb
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error occured while tuning XGBoost model."
                + str(exception_msg),
                "e",
            )
            raise exception_msg

    def get_model_score(
        self, model, model_name, test_x, test_y, log_writer
    ):
        """Get model score

        Args:
            model (class any): Model object
            model_name (str): Model name
            test_x (class 'pandas.core.frame.DataFrame'): Independent Features
            test_y (class 'pandas.core.frame.DataFrame'): Dependent Features
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Returns:
            Float: Model score
        """
        model_pred = model.predict(test_x)
        # If there is only one label in y, accuracy is considered
        # Else, roc_auc_score is considered
        if len(test_y.unique()) == 1:
            model_score = accuracy_score(test_y, model_pred)
            # Log the accuracy score
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Accuracy for {mname}: {score}".format(
                    mname=model_name, score=str(model_score)
                ),
            )
        else:
            # AUC Score
            model_score = roc_auc_score(test_y, model_pred)
            # Log AUC Score
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "AUC for {mname}: {score}".format(
                    mname=model_name, score=str(model_score)
                ),
            )
        return model_score

    def get_best_model(
        self, train_x, train_y, test_x, test_y, log_writer
    ):
        """Get the best model

        Args:
            train_x (class 'pandas.core.frame.DataFrame'):
                            Independent features for training
            train_y (class 'pandas.core.frame.DataFrame'):
                            Target features for training
            test_x (class 'pandas.core.frame.DataFrame'):
                            Independent features for training
            test_y (class 'pandas.core.frame.DataFrame'):
                            Target features for training
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Returns:
            str: Best model name
            int: Best model score
            class any: model object
        """
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Getting the best model",
        )
        all_model_score = dict()
        classifiers = {
            "DTree": "get_best_params_for_dtree",
            "KNN": "get_best_params_for_knn",
            "LogReg": "get_best_params_for_logreg",
            "RandomForest": "get_best_params_for_rforest",
            "SVC": "get_best_params_for_svc",
            "XGBoost": "get_best_params_for_xgboost",
        }

        try:
            for key in classifiers:
                # Call the method to get the best parameters
                model = getattr(self, classifiers[key])(
                    train_x, train_y, log_writer
                )
                # Calculate model score
                model_score = self.get_model_score(
                    model, key, test_x, test_y, log_writer
                )
                # Store the score and model
                all_model_score[key] = [model_score, model]

            # Determining the best model
            best_model_name = max(
                all_model_score, key=all_model_score.get
            )
            return (
                best_model_name,
                all_model_score[best_model_name][0],  # model score
                all_model_score[best_model_name][1],  # model
            )

        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "Error while getting the best model. "
                + str(exception_msg),
            )
