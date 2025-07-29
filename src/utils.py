import pandas as pd
import numpy as np 
import os 
import sys 
import dill 
from src.exception import CustomException 

from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj): 
    try:
        import joblib
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
       
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        model_names = list(models.keys())
        model_objects = list(models.values())  # Get the actual model objects
        
        for i in range(len(models)):
            model_name = model_names[i]
            model_obj = model_objects[i]  # Use values instead of keys
            
            # Fit the model
            model_obj.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model_obj.predict(X_train)
            y_test_pred = model_obj.predict(X_test)
            
            # Calculate scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store the test score in report
            report[model_name] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)

