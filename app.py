from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from textpredict.text_classification_predict import TextClassificationPredict
from common import file
app = Flask(__name__)
CORS(app)

@app.get("/")
def index_get():
    return render_template("base.html") # run interface

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    input = TextClassificationPredict(text,file.get_dbtrain(), file.get_dbtrain_extend(), file.get_dbanswers()) # Class initialization
    model_response = input.Text_Predict() # Text_Predict return a mess
    message = {"answer": model_response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)