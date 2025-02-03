import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import os, pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class DataGenerator:
    def __init__(self, api_key, path_to_data = 'data.csv', categories = ['us_economy', 'sentiment']):
        self.news_api = NewsApiClient(api_key=api_key)
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.columns = ['Title', 'Source', 'Description', 'Label_us_economy', 'Label_sentiment']
        self.path_to_data = path_to_data
        self.categories = categories
        
        # Check if data.csv exists, if not create it with the specified columns
        if not os.path.exists(path_to_data):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv('data.csv')
            print("Created data.csv with columns: title, description, label_us_economy, label_sentiment")

    def classify_and_save(self, articles):

        try:
            existing_data = pd.read_csv(self.path_to_data, index_col='Title')
            existing_titles = existing_data.index.tolist()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            existing_titles = []

        length_data = 0
        classified_data = []
        for article in articles:
            length_data += 1
            title = article.get('title', '')
            description = article.get('description', '')
            source = article.get('source', {}).get('name', '')
            labels = dict()

            skip = False
            stop = False
            if title in existing_titles:
                print(f"Skipping already classified article: {title}")
                for category in self.categories:
                    labels[category] = existing_data.loc[title, f'Label_{category}']
                skip = True
            if not skip:
                print(f"\nTitle: {title}\nDescription: {description}\nSource: {source}")
                for category in self.categories:
                    label = input(f"Classify the sentence (-1, 0, 1) for '{category}' or type 'stop' to end: ")
                    labels[category] = label
                    if label.lower() == 'stop' or stop:
                        stop = True
                        print("Classification stopped by user.")
                        break
                    while label not in ['-1', '0', '1']:
                        print("Invalid input. Please enter -1, 0, 1, or 'stop'.")
                        label = input(f"Classify the sentence (-1, 0, 1) for '{category}' or type 'stop' to end: ")
                        if label.lower() == 'stop' or stop:
                            stop = True
                            print("Classification stopped by user.")
                            break
                    if label.lower() == 'stop' or stop:
                        stop = True
                        break
                if label.lower() == 'stop':
                        break

            existing_titles.append(title)
            if not stop:
                classified_data.append((title, source, description, labels[self.categories[0]], labels[self.categories[1]]))

        df = pd.DataFrame(classified_data, columns=self.columns)
        df.to_csv(self.path_to_data, index=existing_titles)
        print("Data saved to data.csv")

    def get_classified_data(self):
        return pd.read_csv(self.path_to_data)

    
def fetch_financial_news(newsapi_key, query, date):
    """
    Fetches financial news articles based on the given query and date.

    Args:
        query (str): The query string to search for in the news articles.
        date (datetime): The date for which to fetch the news articles.

    Returns:
        list: A list of articles fetched from the news API.
    """
    news_api = NewsApiClient(api_key=newsapi_key)
    date_from = (date - timedelta(days=1)).strftime('%Y-%m-%d')
    all_articles = news_api.get_everything(q=query,
                                    from_param=date_from,
                                    to=date,
                                    language='en',
                                    sort_by='relevancy',
                                    page=5)
    return all_articles['articles']

def preprocess_data(documents):
    stop_words = set(stopwords.words('english'))
    preprocessed_documents = []

    for doc in documents:
        # Tokenize the document
        tokens = word_tokenize(doc)
        # Remove stop words
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        # Join tokens back into a string
        preprocessed_doc = ' '.join(filtered_tokens)
        preprocessed_documents.append(preprocessed_doc)

    return preprocessed_documents

def extract_relevant_sections(text, keywords):
    """
    Extracts relevant sections from the text based on the specified keywords.

    Args:
        text (str): The input text from which to extract relevant sections.
        keywords (list): A list of keywords to search for in the text.

    Returns:
        str: A string containing the relevant sections of the text.
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    relevant_sections = []

    # Search for keywords in each sentence
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            if sentence != '':
                relevant_sections.append(sentence)

    # Join the relevant sentences into a single string
    return ' '.join(relevant_sections)


