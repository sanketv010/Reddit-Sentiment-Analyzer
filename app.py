from flask import Flask, jsonify,request,render_template, redirect, url_for
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline
import random

app = Flask(__name__)

stressed_subreddits = [
    'GetMotivated', 'UpliftingNews', 'KindVoice', 'decidingtobebetter',
    'SelfImprovement', 'CasualConversation', 'LifeProTips', 'selfhelp',
    'BettermentBookClub', 'MentalHealth', 'Mindfulness', 'Anxietyhelp',
    'FeelGood', 'Motivation', 'Support'
]

no_stress_subreddits = [
    'happy', 'MadeMeSmile', 'wholesomememes', 'AnimalsBeingBros', 'Eyebleach',
    'Aww', 'HappyCrowds', 'ContagiousLaughter', 'UpliftingNews',
    'HumansBeingBros', 'Positive', 'GoodNews', 'Joy', 'Calm', 'CheerUp'
]

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    try:
        input_data = request.form['input_data']
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(input_data)

        selected_subreddits = []
        if predictions is not None:
            selected_subreddits = random.sample(
                stressed_subreddits if predictions == 1 else no_stress_subreddits, 5
            )
        #DEBUG
        print("PREDICTIONS:",predictions)
        
        return render_template('index.html',prediction = predictions, selected_subreddits=selected_subreddits)
    
    except Exception as e:
        logging.error(e)
        return render_template('index.html', prediction=None)  
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)            
