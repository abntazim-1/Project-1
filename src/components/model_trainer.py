import os 
import sys 
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor  
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from src.utils import save_object
from src.utils import evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                'CatBoostRegressor': CatBoostRegressor(verbose=False),  # Added verbose=False to reduce output
                'XGBRegressor': XGBRegressor(),
                'LinearRegression': LinearRegression(),
                'RandomForestRegressor': RandomForestRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor()
            }
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # Use evaluate_models function to get initial model comparison
            logging.info("Evaluating all models...")
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            # Get the best model based on R2 score
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")
            
            # Train the best model (it's already trained in evaluate_models, but let's be explicit)
            logging.info(f"Training the best model: {best_model_name}")
            best_model.fit(X_train, y_train)
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model 
            )
            
            # Make final predictions and calculate metrics
            predictions = best_model.predict(X_test)
            r2_value = r2_score(y_test, predictions)
            mae_value = mean_absolute_error(y_test, predictions)
            mse_value = mean_squared_error(y_test, predictions)
            
            logging.info(f"Final model metrics:")
            logging.info(f"R2 Score: {r2_value}")
            logging.info(f"MAE: {mae_value}")
            logging.info(f"MSE: {mse_value}")
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")
            
            return r2_value  
            
        except Exception as e:
            raise CustomException(e, sys)