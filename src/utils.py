import os
import sys
import joblib
from src.exception import CustomException

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)