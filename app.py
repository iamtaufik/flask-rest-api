from flask import Flask, request, jsonify
import pickle
from datetime import datetime

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

def count_month(start_date, end_date):
    # start_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date = datetime.strptime(start_date, "%m/%d/%Y")
    end_date = datetime.strptime(end_date, "%m/%d/%Y")
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    
    if end_date.day < start_date.day:
        months -= 1
    
    return months

@app.route('/')
def home():
    return "Hello World!"

@app.route('/api/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    date_to_predict = request_data['date_to_predict']

    month = count_month("10/31/2019", date_to_predict)
    future = model.make_future_dataframe(periods=month + 1, freq='M')
    forecast = model.predict(future)

    array_of_predict = []
    for i in range(len(forecast['ds'])):
        obj = {
            'date': forecast['ds'][i].strftime("%B %Y"),
            'sales_upper': forecast['yhat_upper'][i],
            'sales': forecast['yhat'][i],
            'sales_lower': forecast['yhat_lower'][i],
        }
        array_of_predict.append(obj)
    
    return jsonify(array_of_predict)

if __name__ == "__main__":
    app.run()