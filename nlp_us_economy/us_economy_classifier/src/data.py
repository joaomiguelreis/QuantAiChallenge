import requests
import os
import pandas as pd
import numpy as np
import csv

class DataGenerator:
    def __init__(self, api_url, api_key, categories = ['us_economy', 'sentiment'], path_to_data = 'data.csv'):
        self.api_url = api_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.columns = ['Title', 'Source', 'Description', 'Label_us_economy', 'Label_sentiment']
        self.path_to_data = path_to_data
        self.categories = categories
        
        # Check if data.csv exists, if not create it with the specified columns
        if not os.path.exists(path_to_data):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv('data.csv')
            print("Created data.csv with columns: title, description, label_us_economy, label_sentiment")

    def fetch_data(self, query):
        response = requests.get(f"{self.api_url}?q={query}", headers=self.headers)
        if response.status_code == 200:
            return response.json().get('articles', [])
        else:
            print("Failed to fetch data")
            return []

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
            