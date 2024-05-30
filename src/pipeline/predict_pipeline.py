import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, input_data):
        try:
            logging.info(input_data)
            model_path=os.path.join('data',"model.pkl")
            preprocessor_path=os.path.join('data_transformation','preprocessor.pkl')
            print("Before Loading")
            #print("PATH_model:",model_path,"PATH_preproc:",preprocessor_path)
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            params = model.get_params()
            #print("Model Parameters:", params)
            print("#####")
            input_data=preprocessor.remove_stopwords(input_data)
            input_data=preprocessor.text_lemmatize(input_data)
            input_data = [input_data]
            preds = model.predict(input_data)
            print("OUTPUT:",preds)
            return preds
        
        except CustomException as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)
