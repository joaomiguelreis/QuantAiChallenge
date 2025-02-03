import pandas as pd

from src.data import DataGenerator

api_url = "https://newsapi.org/v2/everything"
api_key = "3b0b7525822642fc9dba778c9a8e2ac4"  # Replace with your actual API key

generator = DataGenerator(api_url, api_key)
query = "business OR economy OR finance OR politics"
articles = generator.fetch_data(query)
generator.classify_and_save(articles)