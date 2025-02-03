import yaml
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import pandas as pd

from utils.sentiment_analysis import SentimentAnalyser
from utils.data_processing import DataGenerator, preprocess_data, fetch_financial_news
from utils.classifier import Classifier

from transformers import pipeline

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():

    # Load configuration
    config = load_config('sentiment-analysis-tool/config.yaml')

    # Load the custom model
    classifier = Classifier('us_economy_sentiment')
    classifier.load_model('us_economy_sentiment_model.joblib')

    # Initialize sentiment analyzer using NewsAPI client and initialize Hugging Face pipeline
    sentiment_analyzer = SentimentAnalyser(config['api_key'])
    sentiment_pipeline = pipeline('sentiment-analysis')

    # Fetch financial news and analyze sentiment
    query = config['query']

    date = datetime(2025, 1, 25)
    historical_sentiment_hugging_face = []
    historical_sentiment_textBolb = []
    historical_sentiment_classifier = []
    dates = []
    # Loop through each day until today
    while(date < datetime.today()):
        print(date)  # Print the current date
        # Fetch financial news articles for the given query and date
        all_articles = fetch_financial_news(config['api_key'], query, date)

        # Preprocess the articles' descriptions
        prepocessed_articles = preprocess_data([article['description'] for article in all_articles if article['description'] is not None])

        # Analyze the sentiment of the preprocessed articles using Hugging Face
        day_sentiment_hugging_face = sentiment_analyzer.get_day_sentiment_hugging_face(prepocessed_articles, sentiment_pipeline)
        # Analyze the sentiment of the preprocessed articles using TextBlob
        day_sentiment_bold = sentiment_analyzer.get_day_sentiment_textBolb(prepocessed_articles)
        # Analyze the sentiment of the preprocessed articles using the custom classifier
        day_sentiment_classifier = sentiment_analyzer.get_day_sentiment_custom_classifier(prepocessed_articles, classifier)

        print(f"Day Sentiment Hugging Face: {day_sentiment_hugging_face}")  # Print the sentiment from Hugging Face
        print(f"Day Sentiment TextBold: {day_sentiment_bold}")  # Print the sentiment from TextBlob
        print(f"Day Sentiment Classifier: {day_sentiment_classifier}")  # Print the sentiment from the custom classifier

        # Append the current date and sentiments to their respective lists
        dates.append(date)
        historical_sentiment_hugging_face.append(day_sentiment_hugging_face)
        historical_sentiment_textBolb.append(day_sentiment_bold)
        historical_sentiment_classifier.append(day_sentiment_classifier)

        # Move to the next day
        date += timedelta(days=1)
    
    # Convert to DataFrame
    df = pd.DataFrame({'Date': dates, 
                       'Sentiment_HuggingFace': historical_sentiment_hugging_face,
                       'Sentiment_TextBolb': historical_sentiment_textBolb,
                       'Sentiment_Classifier': historical_sentiment_classifier})

    # Handle missing data
    df.fillna(0, inplace=True)  # Fill missing values with 0

    # Save to CSV
    df.to_csv('historical_sentiment.csv', index=False)

    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Sentiment_HuggingFace'], label='Sentiment HuggingFace')
    ax.plot(df['Date'], df['Sentiment_TextBolb'], label='Sentiment TextBolb')
    ax.plot(df['Date'], df['Sentiment_Classifier'], label='Sentiment Classifier')
    ax.set(xlabel='Date', ylabel='Sentiment',
           title='Historical Sentiment')
    ax.grid()
    ax.legend()  # Add legend to the plot
    fig.autofmt_xdate(rotation=45)  # Rotate date labels to 45 degrees

    fig.savefig("sentiment_plot.png")


if __name__ == "__main__":
    main()