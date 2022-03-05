This ML project is an end to end deployment of the detection of credit cards default transactions.

The project uses AWS EC2 instance for the deployment, S3 Bucket to store the necessary files and Apache Cassandra (Astra DB) to store the dataset into the database table.

The front end uses Bootstrap and flask framework.


***************************************************
************** DIRECTORY STRUCTURE **************
***************************************************
│   .env
│   .gitignore
│   common_utility.py
│   data_validation.py
│   foo.txt
│   general_config.json
│   main.py
│   predict_model.py
│   Procfile
│   README.md
│   requirements.txt
│   secure-connect-creditcardfraud.zip
│   train_model.py
│   
├───.vscode
│       launch.json
│       settings.json
│       
├───application_logging
│   │   logger.py
│   │   logger_local_sys.py
│   │   
│   └───__pycache__
│           logger.cpython-39.pyc
│           
├───data
│   ├───bad_prediction_data
│   │       predict_2.csv
│   │       
│   ├───bad_training_data
│   │       UCI_Credit_Card_Dataset.csv
│   │       
│   ├───error_DB_insert_data
│   │       Error Records_14022022_212438.csv
│   │       
│   ├───good_prediction_data
│   ├───good_training_data
│   │       UCI_Credit_Card_Dataset.csv
│   │       
│   ├───prediction_output_file
│   ├───predict_data
│   │       predict.csv
│   │       
│   ├───raw_predict
│   │       predict_2.csv
│   │       
│   ├───raw_training
│   └───training_data
│           Training_Data_14022022_222922.csv
│           
├───DataPreprocess
│   │   clustering.py
│   │   data_preprocess.py
│   │   K-Means_Elbow.PNG
│   │   
│   └───__pycache__
│           clustering.cpython-39.pyc
│           data_preprocess.cpython-39.pyc
│           
├───DataTransformation
│   │   data_transform.py
│   │   
│   └───__pycache__
│           data_transform.cpython-39.pyc
│           
├───DBOperations
│   │   dbOperations.py
│   │   
│   └───__pycache__
│           dbOperations.cpython-39.pyc
│           
├───EDA
│       eda.ipynb
│       
├───log
│   ├───prediction_log
│   ├───prediction_validation_log
│   ├───training_log
│   └───training_validation_log
├───model
│   │   KNN_C_3.sav
│   │   SVC_C_1.sav
│   │   SVC_C_2.sav
│   │   XGBoost_C_0.sav
│   │   
│   └───KMeans
│           KMeans.sav
│           
├───ModelTuner
│   │   model_tuner.py
│   │   
│   └───__pycache__
│           model_tuner.cpython-39.pyc
│           
├───RawDataProccessing
│   │   raw_data_processing.py
│   │   
│   └───__pycache__
│           raw_data_processing.cpython-39.pyc
│           train_raw_data_processing.cpython-39.pyc
│           
├───static
│   │   Backtest.csv
│   │   
│   ├───img
│   │       backimg.jpg
│   │       
│   └───styles
│           Footer-Basic.css
│           Login-Form-Clean-1.css
│           Login-Form-Clean.css
│           Login-Form-Dark.css
│           styles.css
│           
├───templates
│   │   backimg.jpg
│   │   index.html
│   │   ML_credit_card_fraud.bsdesign
│   │   
│   └───assets
│       ├───css
│       └───img
│               backimg.jpg
│               
└───__pycache__
        common_utility.cpython-39.pyc
        data_validation.cpython-39.pyc
        logging.cpython-39.pyc
        main.cpython-39.pyc
        predict_model.cpython-39.pyc
        train_model.cpython-39.pyc
        train_validation.cpython-39.pyc
        
