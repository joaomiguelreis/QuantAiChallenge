import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Download NLTK data files (only need to run once)
nltk.download('stopwords')
nltk.download('punkt')

class Classifier():
    def __init__(self, name, model = None):
        self.name = name
        self.stop_words = set(stopwords.words('english'))
        self.model = None

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

    def train(self, sentences, labels, model):
        self.model = model
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
    
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")