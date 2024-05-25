import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
sys.path.append('src')
from src.exception import CustomException
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from src.logger import logging
nltk.data.path.append('D:\\MLOps Project\\venv\\nltk_data')
import joblib

@dataclass
class DataTransformationConfig:
    root_dir: str=os.path.join('data_transformation')
    train_data_path: str = os.path.join('data', "train.csv")
    test_data_path: str = os.path.join('data', "test.csv")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def text_lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()
        word_list = word_tokenize(text)  
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list]) 
        return lemmatized_output

    def remove_stopwords(self, sentence):
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
                     "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
                     "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
                     "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

        sentence = sentence.lower()
        words = sentence.split()
        no_words = [w for w in words if w not in stopwords]
        sentence = " ".join(no_words)
        return sentence

    def initiate_data_transformation(self):
        logging.info("Data transformation started")
        try:
            os.makedirs(self.data_transformation_config.root_dir, exist_ok=True)
            
            train_df=pd.read_csv(os.path.join(self.data_transformation_config.train_data_path))
            test_df=pd.read_csv(os.path.join(self.data_transformation_config.test_data_path))
            
            logging.info("Train and Test data read")

            
            train_df['text'] = train_df['text'].apply(self.text_lemmatize)
            test_df['text'] = test_df['text'].apply(self.text_lemmatize)
            logging.info("Lemmatization initiated")


            train_df['text'] = train_df['text'].apply(self.remove_stopwords)
            test_df['text'] = test_df['text'].apply(self.remove_stopwords)
            logging.info("Stopwords removed")

            train_df['label_name'] = train_df['label'].map({
                0: 'No stress',
                1: 'Stress'
            })
            test_df['label_name'] = test_df['label'].map({
                0: 'No stress',
                1: 'Stress'
            })
            logging.info("Label Name Added")
            
            train_df.to_csv(os.path.join(self.data_transformation_config.root_dir, "train_processed.csv"), index=False)
            test_df.to_csv(os.path.join(self.data_transformation_config.root_dir, "test_processed.csv"), index=False)  
            
            logging.info("Transformed the data seccessfully")

            joblib.dump(self, os.path.join(self.data_transformation_config.root_dir, "preprocessor.pkl"))
            
            return train_df, test_df, self.data_transformation_config.root_dir
        
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)  
