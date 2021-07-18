
import pandas as pd
from flask import Flask, render_template, url_for, request, redirect
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('cleandata.csv')
pipe = pickle.load(open('XGBModel.pkl', 'rb'))

@app.route('/')


def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations = locations)

@app.route('/predict', methods = ['POST'])
def predict():
    location = request.form.get('location')
    size = float(request.form.get('size'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    input = pd.DataFrame([[location ,size, sqft, bath]],columns=['location', 'size', 'total_sqft', 'bath'])

    prediction = pipe.predict(input)[0] 

    return str(np.round(prediction, 2)) + ' Lakhs'



if __name__ == '__main__':
    app.run(debug = True, port = 5001)