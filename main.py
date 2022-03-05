from flask import Flask, jsonify, render_template, request

from data_validation import DataValidation
from predict_model import PredictModel
from train_model import TrainModel

app = Flask(__name__)


@app.route("/", methods=["GET"])
@app.route("/home", methods=["GET"])
def home():
    return render_template("index.html", flagSuccess=0)


# ----------- T R A I N I N G   S E C T I O N -----------
@app.route("/upload", methods=["POST"])
def uploadTrainData():
    file = request.files["trainFilepath"]
    # values of mode:
    # del_model - Delete existing DB data & insert again & train
    # append_model - Append into exiting DB data & train
    mode = request.form["model_mode"]
    try:
        DataValidation(mode="training").validate_train_data(
            app, file, mode, request.mimetype
        )
        TrainModel().train_model_from_data()
        return jsonify(flagSuccess=1)
    except Exception as exception_msg:  # noqa
        return jsonify(flagSuccess=0)


# ----------- P R E D I C T   S E C T I O N -----------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["predictFilepath"]
    try:
        DataValidation(mode="prediction").validate_predict_data(
            app, file, request.mimetype
        )
        PredictModel().get_prediction_from_data()
        return jsonify(flagSuccess=1)
    except Exception as exception_msg:  # noqa
        return jsonify(flagSuccess=0)


if __name__ == "__main__":
    # app.run()
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=8080)
