import pickle
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify

input_file = 'final_model.bin'

with open(input_file, 'rb') as f_in: 
    model = pickle.load(f_in)

app = Flask('churn')
@app.route('/predict', methods = ['POST'])

def predict():
    customer = request.get_json()
    n_customer = pd.Series(customer)
    y_pred = model.predict_proba(n_customer)[1]
    churn = y_pred >=0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)
