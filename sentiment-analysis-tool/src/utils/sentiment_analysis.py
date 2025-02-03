from newsapi import NewsApiClient
from datetime import datetime, timedelta
from textblob import TextBlob

TODAY = datetime.today().strftime('%Y-%m-%d')
YESTERDAY = datetime.today() - timedelta(days=1)

class SentimentAnalyser:
    def __init__(self, api_key):
        """
        Initializes the SentimentAnalyser with an API client using newsapi package.

        Args:
            api_key (NewsApiClient key)
        """
    
    def get_day_sentiment_textBolb(self, list_of_sentences):
        """
        Analyzes the sentiment of a list of sentences using the TextBlob library.

        Args:
            list_of_sentences (list): A list of sentences to analyze.

        Returns:
            float: The average sentiment for the day, calculated as a weighted average
                   based on the polarity and subjectivity of each sentence.
        """
        average_sentiment = 0
        # Analyze sentiment for each articles
        for sentence in list_of_sentences:
            # Get prediction and prediction probability
            testimonial = TextBlob(sentence)
            
            print(f"Sentence: {testimonial.sentiment}")
            # weighted average of sentiment according to subjectivity
            average_sentiment += testimonial.sentiment.polarity * testimonial.sentiment.subjectivity
        
        if len(list_of_sentences) == 0:
            average_sentiment = 0
        else:
            average_sentiment = average_sentiment / len(list_of_sentences)
        
        return average_sentiment

    def get_day_sentiment_hugging_face(self, list_of_sentences, sentiment_pipeline):
        """
        Analyzes the sentiment of a list of sentences using a Hugging Face sentiment analysis pipeline.

        Args:
            list_of_sentences (list): A list of sentences to analyze.
            sentiment_pipeline (Pipeline): A Hugging Face sentiment analysis pipeline.

        Returns:
            float: The average sentiment for the day, calculated as the average sentiment
                   score of each sentence.
        """
        average_sentiment = 0
        # Analyze sentiment for each articles
        for sentence in list_of_sentences:
            # Get prediction and prediction probability
            prediction = sentiment_pipeline(sentence)[0]
            print(f"Sentence: {sentence}")
            print(f"Prediction: {prediction['label']}")
            print(f"Prediction Probability: {prediction['score']}")
            sentiment = 1 if prediction['label'] == 'POSITIVE' else -1 if prediction['label'] == 'NEGATIVE' else 0
            average_sentiment += sentiment
        
        if len(list_of_sentences) == 0:
            average_sentiment = 0
        else:
            average_sentiment = average_sentiment / len(list_of_sentences)


        return average_sentiment

    def get_day_sentiment_custom_classifier(self, list_of_sentences, classifier):
        """
        Analyzes the sentiment of a list of sentences using a custom classifier.

        Args:
            list_of_sentences (list): A list of sentences to analyze.
            classifier (Classifier): A custom classifier object.

        Returns:
            float: The average sentiment for the day, calculated as the average sentiment
                   score of each sentence.
        """
        average_sentiment = 0
        # Analyze sentiment for each articles
        for sentence in list_of_sentences:
            # Get prediction and prediction probability
            prediction = classifier.predict(sentence)
            print(f"Sentence: {sentence}")
            print(f"Prediction: {prediction}")
            average_sentiment += prediction
        
        if len(list_of_sentences) == 0:
            average_sentiment = 0
        else:
            average_sentiment = average_sentiment / len(list_of_sentences)

        return average_sentiment    
    