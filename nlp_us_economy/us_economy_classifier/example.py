import pandas as pd

from src.classifier import Classifier

import pandas as pd
# from transformers import pipeline



import pandas as pd
from src.classifier import Classifier

data = pd.read_csv('data.csv')
sentences = data['Description'].tolist()
labels_us_economy = data['Label_us_economy'].tolist()
labels_sentiment = data['Label_sentiment'].tolist()

classifier_us_economy = Classifier('us_economy')
classifier_us_economy.train(sentences, labels_us_economy)

# Save the model
classifier_us_economy.save_model('us_economy_model.joblib')

# Load the model
classifier_us_economy.load_model('us_economy_model.joblib')

test_sentence = "US economy is very good"
print(f"Prediction: {classifier_us_economy.predict(test_sentence)}")
print(f"Prediction Probability: {classifier_us_economy.predict_proba(test_sentence)}")