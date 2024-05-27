import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('data',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self, train_df, test_df):
        logging.info("Training model started")

        try:
            X_train = train_df['text']
            y_train = train_df['label']
            X_test = test_df['text']
            y_test = test_df['label']

            xgb_model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
            ])         

            xgb_model.fit(X_train, y_train)

            logging.info("Model trained successfully")

            train_preds = xgb_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_preds)
            logging.info(f"Train Accuracy: {train_accuracy}")

            logging.info("Training Set Evaluation:")
            logging.info(classification_report(y_train, train_preds))
            logging.info("Confusion Matrix:")
            logging.info(confusion_matrix(y_train, train_preds))

            test_preds = xgb_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_preds)

            logging.info(f"Test Accuracy: {test_accuracy}")
            logging.info("Test Set Evaluation:")
            logging.info(classification_report(y_test, test_preds))
            logging.info("Confusion Matrix:")
            logging.info(confusion_matrix(y_test, test_preds))

            return xgb_model

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)
        
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            joblib.dump(model, self.model_trainer_config.trained_model_file_path)

            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Error in saving model: {str(e)}")
            raise CustomException(e, sys)  