from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = {
            'gender': request.form.get('gender'),
            'race/ethnicity': request.form.get('ethnicity'),
            'parental level of education': request.form.get('parental_level_of_education'),
            'lunch': request.form.get('lunch'),
            'test preparation course': request.form.get('test_preparation_course'),
            'reading score': float(request.form.get('reading_score')),
            'writing score': float(request.form.get('writing_score'))
        }
        
        pred_df = pd.DataFrame([data])
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0])
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
    
