from flask import Flask, jsonify,request,render_template
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline

application=Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    try:
        input_data = request.form['input_data']
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(input_data)
        
        return render_template('index.html',prediction = predictions)
    
    except Exception as e:
        return render_template('index.html', prediction=None)  
    
if __name__=="__main__":
    app.run(host="0.0.0.0")            