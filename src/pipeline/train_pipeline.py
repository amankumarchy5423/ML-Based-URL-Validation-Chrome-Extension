import pandas as pd
import numpy as np
import os,sys
from src.logging.logger import my_logger
from src.exception.exception import MyException
from src.component.preprocessing import preprocessing_of_data
from src.component.model_training import Model_Training


class train_pipeline:
    def __init__(self):
        try:
            pass
        except Exception as e :
            my_logger.error(f"Error in train_pipeline class: {str(e)}")
            raise MyException(f"Error in train_pipeline class: {str(e)}",sys)
    
    def preprocessing(self):
        try:
            my_logger.info("preprocessing pipeline started ")
            obj = preprocessing_of_data(data_path='network_data/urldata.csv')
            x , y = obj.initiate_preprocessing()
            return x, y 
            my_logger.info("preprocessing pipeline ended ")
        except Exception as e :
            my_logger.error(f"Error in preprocessing method: {str(e)}")
            raise(e)
    def model_training(self,x_data , y_data):
        try:
            my_logger.info("model training pipeline started ")
            obj = Model_Training(x_data,y_data)
            obj.initiate_model_train()
            my_logger.info("model training pipeline ended ")
        except Exception as e :
            my_logger.error(f"Error in model_training method: {str(e)}")
            raise(e)
    
    def initiate_train_pipeline(self):
        try:
            my_logger.info("train pipeline initiated ")
            x,y = self.preprocessing()
            self.model_training(x,y)
            my_logger.info("train pipeline ended ")
        except Exception as e :
            my_logger.error(f"Error in initiate_train_pipeline method: {str(e)}")
            raise(e)
        



