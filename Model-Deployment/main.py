import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, request, jsonify

model = keras.models.load_model("best_model.h5")

def prepare_data(id_num, tipe_kendaraan, jenis_kendaraan, business):
    tipe_kendaraan_dict = {
        'Mobil': 0,
        'Motor': 1
    }
    jenis_kendaraan_dict = {
        'suv': 5,
        'matic': 2,
        'mpv': 3,
        'hatchback': 1,
        'van': 6,
        'bebek': 0,
        'sedan': 4
    }
    business_dict = {
        'Liburan/Weekend': 1,
        'Daily Activity': 0,
        'Travel': 2
    }

    tipe_kendaraan_ = tipe_kendaraan_dict.get(tipe_kendaraan)
    jenis_kendaraan_ = jenis_kendaraan_dict.get(jenis_kendaraan)
    business_ = business_dict.get(business)
    data = [id_num, tipe_kendaraan_, jenis_kendaraan_, business_]
    features = np.array(data)
    return features

def predict(x):
    predictions = model.predict(np.expand_dims(x, axis=0))
    labels = ['< 100.000', '100.000 - 200.000', '200.000 - 500.000', '> 500.000']
    label0 = np.argmax(predictions)
    predicted_label = labels[label0]
    return predicted_label

app = Flask(__name__)

@app.route("/", methods=["POST"])
def index():
    if request.method == "POST":
        try:
            input_data = request.get_json()
            if not input_data:
                return jsonify({"error": "no data"})

            feature1 = input_data.get('feature1')
            feature2 = input_data.get('feature2')
            feature3 = input_data.get('feature3')
            feature4 = input_data.get('feature4')
            features = prepare_data(feature1, feature2, feature3, feature4)
            prediction = predict(features)

            data = {"prediction": prediction}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)
