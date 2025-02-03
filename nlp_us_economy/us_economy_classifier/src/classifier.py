# filepath: us_economy_classifier/us_economy_classifier.py
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from src.data import DataGenerator

# Download NLTK data files (only need to run once)
nltk.download('stopwords')
nltk.download('punkt')

class Classifier(DataGenerator):
    def __init__(self, name):
        self.name = name
        self.stop_words = set(stopwords.words('english'))
        self.model = make_pipeline(TfidfVectorizer(preprocessor=self.preprocess_text), LogisticRegression())

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")

    def preprocess_text(self, text):
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Convert to lower case
        tokens = [token.lower() for token in tokens]
        # Remove punctuation
        tokens = [token for token in tokens if token.isalnum()]
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        # Join tokens back into a single string
        return ' '.join(tokens)

    def train(self, sentences, labels):
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy}")

    def predict(self, sentence):
        prediction = self.model.predict([sentence])[0]
        return prediction

    def predict_proba(self, sentence):
        proba = self.model.predict_proba([sentence])[0]
        return proba
    
    def train_better(self, sentences, labels):
        '''
        Creating a pipeline with a TF-IDF vectorizer and an SVM classifier involves chaining together these two 
        steps into a single, streamlined process. This allows for efficient data transformation and model training. 
        Here's an explanation of each component and how they work together in a pipeline:

        TF-IDF Vectorizer
         - TF-IDF (Term Frequency-Inverse Document Frequency): This is a statistical measure used to evaluate the 
         importance of a word in a document relative to a collection of documents (corpus). It helps in converting 
         text data into numerical features that can be used by machine learning algorithms.
            . Term Frequency (TF): Measures how frequently a term appears in a document.
            . Inverse Document Frequency (IDF): Measures how important a term is by considering how common or rare 
            it is across all documents in the corpus.
        SVM Classifier
         - SVM (Support Vector Machine): This is a supervised machine learning algorithm used for classification 
         tasks. It works by finding the hyperplane that best separates the data into different classes. SVM is 
         effective in high-dimensional spaces and is versatile due to the use of different kernel functions.
        Pipeline
         - Pipeline: This is a tool provided by scikit-learn that allows you to chain multiple processing steps 
         together. Each step in the pipeline is a tuple consisting of a name and an estimator (which can be a 
         transformer or a model). The pipeline ensures that the steps are executed in sequence, and it allows 
         for easy cross-validation and parameter tuning.
        '''

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

        # Create a pipeline with a TF-IDF vectorizer and an SVM classifier
        pipeline = make_pipeline(TfidfVectorizer(preprocessor=self.preprocess_text), SVC(probability=True))

        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = pipeline.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy}")

        # Update the model attribute with the trained pipeline
        self.model = pipeline

        # Define hyperparameters to tune
        parameters = {
            'tfidfvectorizer__max_df': [0.8, 0.9, 1.0],
            'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['linear', 'rbf']
        }

        # Perform grid search to find the best hyperparameters
        grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
