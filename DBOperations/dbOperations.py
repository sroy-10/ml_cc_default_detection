import os
from datetime import datetime
from inspect import currentframe, getframeinfo
from os import path

import pandas as pd
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile
from dotenv import load_dotenv


class DbOperations:
    """This class is to perform all the database operations"""

    def __init__(self):
        # Load environmental variables
        load_dotenv()
        self.FILE_BASENAME = path.basename(__file__)
        # Load DB connection related parameters
        self.ASTRA_CLIENT_ID = os.getenv("ASTRA_CLIENT_ID")
        self.ASTRA_PATH_TO_SECURE_BUNDLE = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                os.getenv("ASTRA_PATH_TO_SECURE_BUNDLE"),
            )
        )
        self.ASTRA_CLIENT_SECRET = os.getenv("ASTRA_CLIENT_SECRET")

        # Determines the timeout of the CQL
        self.REQUEST_TIMEOUT = os.getenv("REQUEST_TIMEOUT")
        # Determines how many times it will try to connect the cluster
        self.AUTO_RETRY = os.getenv("AUTO_RETRY")

    def modifyColumnName(
        self,
        json_datacol_list=list(),
        db_col_list=list(),
        toDb_flag="",
        fromDb_flag="",
    ):
        """Modify column name to make it DB compatible

        Args:
            json_datacol_list (list, optional): JSON column list
            db_col_list ([type], optional): DB column list
            toDb_flag (str, optional): Flag to modify column name as
                                       per DB standards. Defaults to "".
            fromDb_flag (str, optional): Flag to modify column name as
                                       per JSON standards. Defaults to "".

        Returns:
            str: Column name list
        """
        col_name_list = list()
        if len(json_datacol_list) == 0 and len(db_col_list) == 0:
            return col_name_list

        # Making the column name cassandra compatible
        if toDb_flag == "X":
            for json_datacol in json_datacol_list:
                # Replace the characters and convert to lower case
                json_datacol = (
                    json_datacol.replace(" ", "_")
                    .replace(".", "_")
                    .replace("-", "_")
                ).lower()
                col_name_list.append(json_datacol)

        # Reverting the cassandra column name to original column name
        if fromDb_flag == "X":
            for db_col in db_col_list:
                # For serial number, skip checking with JSON
                if db_col == "sno":
                    col_name_list.append("sno")
                    continue
                db_col_split = db_col.split("_")
                db_col_word_len = len(db_col_split)
                flag = 0
                for json_datacol in json_datacol_list:
                    # Check if all the DB col words match with JSON col
                    for i in range(db_col_word_len):
                        if (
                            db_col_split[i].lower()
                            in json_datacol.lower()
                        ):
                            flag = 1
                        else:
                            if flag == 1 or i == 0:
                                break
                    if flag == 1:
                        col_name_list.append(json_datacol)
                        break
        return col_name_list

    def getDBSession(self, log_writer):
        """Create cassandra DB session

        Args:
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            class 'cassandra.cluster.Session': Cassandra session
        """
        execution_profil = ExecutionProfile(
            request_timeout=self.REQUEST_TIMEOUT
        )
        profiles = {"node1": execution_profil}
        cloud_config = {
            "secure_connect_bundle": self.ASTRA_PATH_TO_SECURE_BUNDLE
        }
        auth_provider = PlainTextAuthProvider(
            self.ASTRA_CLIENT_ID, self.ASTRA_CLIENT_SECRET
        )
        cluster = Cluster(
            cloud=cloud_config,
            auth_provider=auth_provider,
            execution_profiles=profiles,
        )
        try:
            # If the session is not connected at first instance,
            # try to auto-connect
            try:
                session = cluster.connect()
            except Exception as exception_msg:
                # If any exception occurs at the last attempt, raise it
                # Else, continue re-attempting
                for i in range(self.AUTO_RETRY):
                    session = cluster.connect()
                    if i == self.AUTO_RETRY - 1:
                        raise exception_msg
                    else:
                        continue
            # If the session connects, log it
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "DB Table connection established",
            )
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                str(exception_msg),
                "e",
            )
            raise exception_msg
        return session

    def getDBTableColName(self, table_name, log_writer):
        """Get DB column names

        Args:
            table_name (str): DB Table Name
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            list: List of columns
        """
        col_name = list()
        try:
            query = (
                "SELECT * FROM system_schema.columns "
                + "WHERE keyspace_name = 'creditcardfraud' AND "
                + "table_name = '{tablename}';"
            ).format(tablename=table_name)
            session = self.getDBSession(log_writer)
            # Store column name
            df = pd.DataFrame(list(session.execute(query)))
            col_name = df["column_name"].values.tolist()
            col_name = [x.upper() for x in col_name]
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                str(exception_msg),
                "e",
            )
            raise exception_msg
        finally:
            session.shutdown()
            return col_name

    def checkTableExist(self, table_name, log_writer):
        """Check if DB table exist or not

        Args:
            table_name (str): DB table name
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Returns:
            bool: True - If BD table exist else False
        """
        try:
            query = (
                "SELECT table_name FROM system_schema.tables "
                + "WHERE keyspace_name='creditcardfraud'"
                + ";"
            )
            session = self.getDBSession(log_writer)
            # Store all the DB table name into the pandas dataframe
            df = pd.DataFrame(list(session.execute(query)))
            if table_name in list(df.table_name):
                log_writer.log(
                    self.FILE_BASENAME,
                    getframeinfo(currentframe()).lineno,
                    "Table {table_name} already exist in DB".format(
                        table_name=table_name
                    ),
                )
                return True
            else:
                return False
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                str(exception_msg),
                "e",
            )
            raise exception_msg
        finally:
            session.shutdown()

    def deleteTable(self, table_name, log_writer):
        """Delete existing table

        Args:
            table_name (str): DB table name
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Returns:
            exception_msg: Captures the exception messages
        """
        msg = ""
        try:
            session = self.getDBSession(log_writer)
            query = (
                'DROP TABLE IF EXISTS "creditcardfraud".{tablename};'
            ).format(tablename=table_name)
            # Execute the query to delete table
            session.execute(query)

        except Exception as exception_msg:
            msg = str(exception_msg)

        finally:
            session.shutdown()
            # Check if the table has been deleted properly or not
            if not self.checkTableExist(table_name, log_writer):
                log_writer.log(
                    self.FILE_BASENAME,
                    getframeinfo(currentframe()).lineno,
                    "Table {table_name} deleted from DB".format(
                        table_name=table_name
                    ),
                )
            else:
                log_writer.log(
                    self.FILE_BASENAME,
                    getframeinfo(currentframe()).lineno,
                    (
                        "Table {table_name} has not been deleted from DB. "
                        + "{e}"
                    ).format(table_name=table_name, e=msg),
                    "e",
                )
                raise Exception("Table can not be deleted." + str(msg))

    def createTable(self, table_name, json_data, log_writer):
        """Create DB table

        Args:
            table_name (str): DB table name
            json_data (dict): JSON Data
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages
        """
        msg = ""
        colnam_dt = create_st = alter_st = ""
        # Get DB session
        session = self.getDBSession(log_writer)
        # Check if table exist or not
        if self.checkTableExist(table_name, log_writer):
            raise Exception(
                ("Table {tabname} already created in DB").format(
                    tabname=table_name
                )
            )

        try:
            for key in json_data.keys():
                # Converting the datatype to Cassandra equivalent
                if (
                    json_data[key] == "Integer"
                    or json_data[key] == "Integers"
                ):
                    data_type = "int"
                elif json_data[key] == "text":
                    data_type = "string"
                else:
                    data_type = json_data[key]
                # Column name change
                col_name = (
                    key.replace(" ", "_")
                    .replace(".", "_")
                    .replace("-", "_")
                )
                if len(create_st) == 0:
                    create_st = (
                        "CREATE TABLE creditcardfraud."
                        + '"{table_name}"'
                        + "(sno int primary key, {column_name} "
                        + "{data_type});"
                    ).format(
                        table_name=table_name,
                        column_name=col_name,
                        data_type=data_type,
                    )
                else:
                    if len(colnam_dt) == 0:
                        colnam_dt = col_name + " " + data_type
                    else:
                        colnam_dt = (
                            colnam_dt
                            + ", "
                            + col_name
                            + " "
                            + data_type
                        )
            # Creating ALTER statement
            alter_st = (
                "ALTER TABLE creditcardfraud.{table_name} ADD ({col});"
            ).format(table_name=table_name, col=colnam_dt)

            # --------------- Create Table ---------------
            try:
                msg = ""
                # Execute Query
                session.execute(create_st)
            except Exception as exception_msg:
                msg = str(exception_msg)
            finally:
                if not self.checkTableExist(table_name, log_writer):
                    log_writer.log(
                        self.FILE_BASENAME,
                        getframeinfo(currentframe()).lineno,
                        (
                            "Error in creating DB table: "
                            + "{table_name}. {e}"
                        ).format(
                            table_name=table_name,
                            e=msg,
                        ),
                        "e",
                    )
                    raise Exception(
                        "Table can not be created." + str(msg)
                    )
                else:
                    log_writer.log(
                        self.FILE_BASENAME,
                        getframeinfo(currentframe()).lineno,
                        'DB Table "{table_name}" created'.format(
                            table_name=table_name
                        ),
                        "s",
                    )

            # --------------- Alter Table ---------------
            try:
                msg = ""
                # Execute Query
                session.execute(alter_st)

            except Exception as exception_msg:
                msg = str(exception_msg)

            finally:
                # if the column has been added, then its a success
                db_col_name = self.getDBTableColName(
                    table_name, log_writer
                )
                if len(db_col_name) == len(json_data.keys()) + 1:
                    # Log
                    log_writer.log(
                        self.FILE_BASENAME,
                        getframeinfo(currentframe()).lineno,
                        "Columns added to the DB table",
                        "s",
                    )
                else:
                    log_writer.log(
                        self.FILE_BASENAME,
                        getframeinfo(currentframe()).lineno,
                        (
                            "Error in creating DB table: "
                            + "{table_name}. {e}"
                        ).format(
                            table_name=table_name,
                            e=msg,
                        ),
                        "e",
                    )
                    raise Exception(
                        "Table can not be created." + str(msg)
                    )

        except Exception as exception_msg:
            msg = str(exception_msg)

        finally:
            session.shutdown()
            if len(msg) != 0:
                log_writer.log(
                    self.FILE_BASENAME,
                    getframeinfo(currentframe()).lineno,
                    "Error in creating DB table: {table_name}. {e}".format(
                        table_name=table_name, e=msg
                    ),
                    "e",
                )
                raise Exception(msg)

    def insertTableRec(
        self, table_name, json_datacol_detail, data, log_writer
    ):
        """Insert records to DB

        Args:
            table_name (str): DB Table name
            json_datacol_detail (dict): JSON column details
            data (class 'pandas.core.frame.DataFrame'): Dataset
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages
        """
        # Dynamically fetch the column name to create the insert query
        # The first column is 'SNO' (Primary key)
        # To store column names as per DB standards
        FLAG_NO_REC_INSERT = 0
        col_name_db = '"sno"'
        col_name_q = "?"
        col_name_list = list(json_datacol_detail.keys())

        # Replace the characters and convert to lower case
        col_name_list_db = self.modifyColumnName(
            json_datacol_list=col_name_list, toDb_flag="X"
        )

        for idx, col in enumerate(col_name_list):
            # Converting the data into Cassandra equivalent types
            if (
                json_datacol_detail[col] == "Integer"
                or json_datacol_detail[col] == "Integers"
            ):
                col_data = list(map(lambda x: int(x), data[col]))
                data[col] = col_data

            # Preparing the column list for the prepare query
            col_name_db = (
                col_name_db + ', "' + col_name_list_db[idx] + '"'
            )
            col_name_q = col_name_q + ", ?"

        # Connect to the session and insert record to DB
        session = self.getDBSession(log_writer)
        # -->Get the last serial number of the record
        query = (
            "select max(sno) as lastsno from "
            + "creditcardfraud.{table_name};"
        ).format(table_name=table_name)

        result_set = pd.DataFrame(list(session.execute(query)))
        last_sno = result_set.iloc[0]["lastsno"]
        if last_sno is None:
            last_sno = 1

        # -->Insert record into database
        query = (
            "INSERT INTO creditcardfraud.{table_name} "
            + "({col}) VALUES ({que})"
        ).format(col=col_name_db, que=col_name_q, table_name=table_name)
        # DataFrame to store error records
        error_data = pd.DataFrame(columns=data.columns)
        try:
            prepared = session.prepare(query)
            data_list = data.values.tolist()
            for row in data_list:
                try:
                    row[0] = last_sno
                    # Prepare query statement
                    bound = prepared.bind(row)
                    # Execute the statement
                    session.execute(bound)
                    # Increment the serial number
                    last_sno = last_sno + 1
                except Exception as exception_msg:
                    # Collect the error records
                    error_data = error_data.append(
                        pd.Series(row, index=error_data.columns),
                        ignore_index=True,
                    )
                    # Log
                    log_writer.log(
                        self.FILE_BASENAME,
                        getframeinfo(currentframe()).lineno,
                        str(exception_msg),
                        "e",
                    )

        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                str(exception_msg),
                "e",
            )
            FLAG_NO_REC_INSERT = 1
        finally:
            session.shutdown()
            return FLAG_NO_REC_INSERT, error_data

    def getDbDataToCsv(
        self, table_name, json_datacol_detail, log_writer
    ):
        """Get DB Data to CSV

        Args:
            table_name (str): DB Table name
            json_datacol_detail (dict): JSON column details
            log_writer (class 'application_logging.logger.AppLogger'):
                            Object for logging

        Raises:
            exception_msg: Captures the exception messages

        Returns:
            str: filename of the saved file
        """
        json_datacol_list = list(json_datacol_detail.keys())
        # Modify the column name as per cassandra standards
        col_name_list_db = self.modifyColumnName(
            json_datacol_list=json_datacol_list,
            toDb_flag="X",
        )
        query = (
            "select sno, {col_name} from creditcardfraud."
            + "{table_name};"
        ).format(
            col_name=(", ".join(col_name_list_db)).lower(),
            table_name=table_name,
        )
        log_writer.log(
            self.FILE_BASENAME,
            getframeinfo(currentframe()).lineno,
            "Export of DB Data to CSV started",
        )
        result_df = pd.DataFrame()
        try:
            current_timestamp = datetime.now()
            FILENAME = "Training_Data_{dt}_{ti}.csv".format(
                dt=str(current_timestamp.strftime("%d%m%Y")),
                ti=str(current_timestamp.strftime("%H%M%S")),
            )
            session = self.getDBSession(log_writer)
            # Get the data from DB
            result_df = pd.DataFrame(list(session.execute(query)))

            # Modify the column name and arrange the data according to JSON
            col_list_rename = self.modifyColumnName(
                json_datacol_list=list(json_datacol_detail.keys()),
                db_col_list=list(result_df.keys()),
                fromDb_flag="X",
            )
            result_df.columns = col_list_rename
            result_df.sort_values(by=["sno"], inplace=True)

            # Save the data to folder
            # result_df.to_csv(full_filepath, index=False)

            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                "DB data exported successfully: "
                + ". Data Shape: "
                + str(result_df.shape),
                "s",
            )
        except Exception as exception_msg:
            log_writer.log(
                self.FILE_BASENAME,
                getframeinfo(currentframe()).lineno,
                str(exception_msg),
                "e",
            )
            raise exception_msg
        finally:
            session.shutdown()
            return FILENAME, result_df
