# Sentiment Analysis Tool

This project is a sentiment analysis tool designed to analyze financial news articles, research reports, and earnings call transcripts, specifically focusing on "recession fears" in the US equity markets. It utilizes two LLM - Hugging Face and TextBolb API - as well as a custom sentiment classifier designed by myself. This code generates sentiment scores that help in understanding market sentiments related to economic downturns.

## Features

- Analyze sentiment of financial documents related to recession fears.
- Process and clean input data for accurate analysis.
- Communicate with a large language model API for sentiment scoring.

## Project Structure

```
sentiment-analysis-tool
├── src
│   ├── main.py                # Entry point of the application
│   ├── classify_data.py       # To classify the data used to train the custom classifier
│   └── utils
│       ├── __init__.py        # Utility functions and constants
│       ├── sentiment_analysis.py  # Sentiment analysis utilities
│       ├── data_processing.py     # Data processing utilities
│       └── classifier.py          # Custom classifier utilities
├── requirements.txt            # Project dependencies
├── config.yaml                 # Configuration settings
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd sentiment-analysis-tool
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the application by editing the `config.yaml` file with your API keys and settings.

## Classify data

The custom classifier is trained using human classified data. To classify the data, excecute the following command:

```
python src/classify_data.py
```

and follow the instructions in the terminal. Once the classification is done, the classifier is stored under the name us_economy_sentiment_model.joblib. A file named 'data.csv' is generated with the trained data

## Usage

To run the sentiment analysis tool, execute the following command:

```
python src/main.py
```

You need to generate a build classifier first

## Output
A plot with the sentiment over time predicted by the three models: Hugging Face, TextBolb and the custom classifier

## Acknowledges
I could not have done this without CoPilot. If only I had this tool during my Ph.D....