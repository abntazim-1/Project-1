import pandas as pd
import numpy as np 
import os 
import sys 
import dill 
from src.exception import CustomException 

from src.logger import logging


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
    