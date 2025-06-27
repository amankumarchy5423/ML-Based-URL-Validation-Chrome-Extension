import pandas as pd
import numpy as np
import os,sys
from src.logging.logger import my_logger
from src.exception.exception import MyException
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib



class preprocessing_of_data:
    def __init__(self, data_path :str = 'network_data/urldata.csv'):
        try:
            self.data_path = data_path
            self.data = pd.read_csv(data_path)
        except Exception as e:
            my_logger.error(f"Error in preprocessing_of_data class: {str(e)}")
            raise MyException(f"Error in preprocessing_of_data class: {str(e)}",sys)
    
    def impute_missing_values(self,df):
        try:
            imputer = SimpleImputer(strategy='most_frequent')
            df[df.columns] = imputer.fit_transform(df)
            return pd.DataFrame(df, columns=df.columns)
        except Exception as e:
            my_logger.error(f"Error in impute_missing_values method: {str(e)}")
            raise MyException(f"Error in impute_missing_values method: {str(e)}",sys)
        

    def encode_categorical(self,df, categorical_columns = 'label'):
        try:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(df[categorical_columns])
            return encoded
        except Exception as e:
            my_logger.error(f"Error in encode_categorical method: {str(e)}")
            raise MyException(f"Error in encode_categorical method: {str(e)}",sys)
    
    
    
    def remove_duplicates(self,df : pd.DataFrame):
        try:
            df = df.drop_duplicates()
            return pd.DataFrame(df)
        except Exception as e:
            my_logger.error(f"Error in remove_duplicates method: {str(e)}")
            raise MyException(f"Error in remove_duplicates method: {str(e)}",sys)
    
   
    def drop_high_na_columns(df : pd.DataFrame, threshold : float = 0.5):
        na_ratio = df.isnull().mean()
        return pd.DataFrame(df.drop(columns=na_ratio[na_ratio > threshold].index))
    
    def save_in_local(self,df):
        try:
            os.makedirs('network_data',exist_ok=True)
            df.to_csv('network_data/data.csv', index=False)
            return 'network_data/data.csv'
        except Exception as e:
            my_logger.error(f"Error in save_in_local method: {str(e)}")
            raise MyException(f"Error in save_in_local method: {str(e)}",sys)
        
    def text_vectoriztion(self,df):
        try:
            df['url'] = df['url'].astype(str)

            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))  # character n-grams work well for URLs
            X = vectorizer.fit_transform(df['url'])

            file_name = "best_model/preprocessor.joblib"
            os.makedirs(os.path.dirname(file_name),exist_ok=True)
            joblib.dump(vectorizer,filename=file_name)
            my_logger.info(f"{X}")
            return X
        except Exception as e:
            my_logger.error(f"Error in text_vectoriztion method: {str(e)}")
            raise MyException(f"Error in text_vectoriztion method: {str(e)}",sys)

    
    def initiate_preprocessing(self):
        try:
            my_logger.info("data preprocessing initiated")

            df = pd.read_csv('network_data/urldata.csv')
            df1 = self.remove_duplicates(df)
            my_logger.info("duplicate data deleted")

            df3 = self.impute_missing_values(df1)
            my_logger.info("missing values imputed")

            df5  = self.remove_duplicates(df3)
            my_logger.info("duplicate data deleted")

            X = self.text_vectoriztion(df5)

            Y = self.encode_categorical(df5)
            my_logger.info("categorical data encoded")

            path = self.save_in_local(df5)
            my_logger.info("data preprocessing completed")


            return X , Y
        except Exception as e:
            my_logger.error(f"Error in initiate_preprocessing method: {str(e)}")
            raise MyException(f"Error in initiate_preprocessing method: {str(e)}",sys)