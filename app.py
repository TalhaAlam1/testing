import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/api', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Empty request"}), 400
            features = [np.array(list(data.values()))]
            prediction = model.predict(features)
            output = float(prediction[0])  # Convert np.float32 to standard float
            return jsonify(output)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:  # GET request
        return "This is the API endpoint. Please use POST method to get predictions."

if __name__ == '__main__':
    app.run(port=5000, debug=True)
