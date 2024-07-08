from flask import Flask, request, jsonify
import pickle
from datetime import datetime
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

model = pickle.load(open('trained_model.pkl', 'rb'))

def format_date(date):
    day = date.day
    month = date.strftime("%B")
    year = date.year
    return f"{day} {month} {year}"

@app.route('/')
def home():
    return "Hello World!"

@app.route('/api/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    date_to_predict = request_data['date_to_predict']

    dates = datetime.strptime(date_to_predict, "%m/%d/%Y")

    future_dates = pd.DataFrame({'ds': pd.date_range(start='10/31/2019', end=dates, freq='D')})

    forecast = model.predict(future_dates)

    array_of_predict = []
    for i in range(len(forecast)):
        obj = {
            'date': format_date(forecast['ds'].iloc[i]),
            'sales_upper': round(forecast['yhat_upper'].iloc[i]),
            'sales': round(forecast['yhat'].iloc[i]),
            'sales_lower': round(forecast['yhat_lower'].iloc[i]),
        }
        array_of_predict.append(obj)

    return jsonify(array_of_predict)

if __name__ == "__main__":
    app.run()