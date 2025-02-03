# Sentiment Analysis Tool

This project is a sentiment analysis tool designed to analyze financial news articles, research reports, and earnings call transcripts, specifically focusing on "recession fears" in the US equity markets. It utilizes a large language model API to generate sentiment scores that help in understanding market sentiments related to economic downturns.

## Features

- Analyze sentiment of financial documents related to recession fears.
- Process and clean input data for accurate analysis.
- Communicate with a large language model API for sentiment scoring.

## Project Structure

```
sentiment-analysis-tool
├── src
│   ├── main.py                # Entry point of the application
│   ├── sentiment_analysis.py   # Contains the SentimentAnalyzer class
│   ├── data_processing.py      # Functions for data processing and cleaning
│   ├── api_client.py           # Handles API communication
│   └── utils
│       └── __init__.py        # Utility functions and constants
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

## Usage

To run the sentiment analysis tool, execute the following command:

```
python src/main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.