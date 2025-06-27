import pandas as pd
import numpy as np
import os,sys
from src.logging.logger import my_logger
from src.exception.exception import MyException
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib





class Model_Training:
    def __init__(self,X_data, Y_data):
        try:
            self.x_data = X_data
            self.target = Y_data
        except Exception as e:
            my_logger.error(f"An error occurred: {str(e)}")
            raise MyException(f"An error occurred: {str(e)}",sys)
        
    
        
    def train_model(self,models : set):
        try:

            my_logger.info(f"y data types : {type(self.target)}")
            X_train, X_test, y_train, y_test = train_test_split(self.x_data, self.target, test_size=0.2)
            best_model = None
            best_score = 0
            best_model_name = ""
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                my_logger.info(f"{name}: Accuracy = {acc:.4f}")
                
                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_model_name = name
            my_logger.info(f"best ML model is {best_model_name} with accuracy {best_score:.4f}")
            
            preprocessor = joblib.load("best_model/preprocessor.joblib")

            ML_model = Pipeline([
                ("preprocessor", preprocessor),
                ("best_model", best_model)
            ])

            joblib.dump(ML_model,filename="best_model/ML_model.joblib")

        except Exception as e:
            my_logger.error(f"An error occurred: {str(e)}")
            raise MyException(f"An error occurred: {str(e)}",sys)
    
    def initiate_model_train(self):
        try:
            # df = pd.read_csv('network_data/urldata.csv')
            my_logger.info("data loaded....")

            models = {
            "Logistic Regression": LogisticRegression(max_iter=1000,verbose=1),
            # "Decision Tree": DecisionTreeClassifier(),
            # "Random Forest": RandomForestClassifier(verbose=1),
            # "SVM": SVC(probability=True,verbose=1),
            # "KNN": KNeighborsClassifier(),
            # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            }
            my_logger.info("set of all classier")

            self.train_model(models=models)

        except Exception as e:
            my_logger.error(f"An error occurred: {str(e)}")
            raise MyException(f"An error occurred: {str(e)}",sys)


