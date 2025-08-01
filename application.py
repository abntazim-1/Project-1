
import pickle
from flask import Flask, request, render_template, jsonify 
import numpy as np 
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from src.logger import logging
import sys 


  
application = Flask(__name__)

app = application 

#Route for home page 

@app.route('/') 
def index():
    return render_template('index.html') 

@app.route("/predictdata", methods=['GET' ,'POST'])
def predict_datapoint():
    try :
        if request.method == 'GET':
            return render_template('home.html')
        
        else:
            logging.info("Received POST request with form data")
            # Extracting form data
            data = CustomData( 
                    gender = request.form.get("gender"),
                    race_ethnicity= request.form.get("race_ethnicity"),
                    parental_level_of_education = request.form.get("parental_level_of_education"), 
                    lunch = request.form.get("lunch"),
                    test_preparation_course = request.form.get("test_preparation_course"),
                    writing_score= float(request.form.get("writing_score")),
                    reading_score = float(request.form.get("reading_score"))
                    )
            pred_data = data.get_data_as_data_frame()
            predict_pipeline = PredictionPipeline() 
            results = predict_pipeline.predict(features=pred_data)
            return render_template('home.html', results=results[0] if results else "No prediction made")  

    
    except Exception as e:
        raise CustomException(e, sys) 
    
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")  
                 
                          
    
    