from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('model_new.pkl', 'rb'))

@app.route('/api/predict')
def predict():
    future = model.make_future_dataframe(periods=12, freq='M')
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