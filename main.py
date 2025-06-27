from src.exception.exception import MyException
from src.logging.logger import my_logger
import sys
import os
import pandas as pd
from src.pipeline.train_pipeline import train_pipeline



my_logger.info("<<<<< main.py >>>>>>")

obj = train_pipeline()
obj.initiate_train_pipeline()

my_logger.info("<<<<< main.py >>>>>>")


        
