import pandas as pd
import yaml
from datetime import datetime

from utils.data_processing import DataGenerator, fetch_financial_news
from utils.classifier import Classifier

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():

    query = "Economy"

    # Load configuration
    config = load_config('sentiment-analysis-tool/config.yaml')

    # Fetch financial news articles
    articles = fetch_financial_news(config['api_key'],query, datetime.today())

    # Generate and save the data
    generator = DataGenerator(config['api_key'], config['path_to_data'])
    generator.classify_and_save(articles)

    # Load the data
    data = pd.read_csv(config['path_to_data'])
    sentences = data['Description'].tolist()
    labels_sentiment = data['Label_sentiment'].tolist()

    # intitate and train classifier
    classifier = Classifier('us_economy_sentiment')
    model = make_pipeline(TfidfVectorizer(preprocessor=classifier.preprocess_text), LogisticRegression())
    classifier.train(sentences, labels_sentiment, model)


    # Save the model
    classifier.save_model('us_economy_sentiment_model.joblib')

if __name__ == "__main__":
    main()