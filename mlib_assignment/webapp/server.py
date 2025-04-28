from flask import Flask, request, jsonify, send_from_directory
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql import SparkSession
import findspark
import os

findspark.init()

spark = SparkSession.builder.appName("LyricsGenrePrediction").getOrCreate()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/mendeley_model")
INDEXER_PATH = os.path.join(BASE_DIR, "../models/label_indexer")

model = PipelineModel.load(MODEL_PATH)
label_indexer = StringIndexerModel.load(INDEXER_PATH)

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/app.js')
def js():
    return send_from_directory(BASE_DIR, 'app.js')

@app.route('/static/<path:path>')
def static_file(path):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), path)

@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    lyrics = content.get("lyrics", "")
    input_df = spark.createDataFrame([(lyrics,)], ["lyrics"])
    prediction = model.transform(input_df)
    prob_vector = prediction.select("probability").collect()[0][0]
    labels = label_indexer.labels
    response = {
        "labels": labels,
        "probabilities": [round(prob_vector[i], 4) for i in range(len(labels))]
    }
    print("Prediction response:", response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)
