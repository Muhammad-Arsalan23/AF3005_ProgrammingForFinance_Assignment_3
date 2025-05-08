# SmartFin Analyzer Plus

An enhanced financial machine learning tool built with Streamlit that provides various analysis capabilities for stock market data.

## Features

- 📈 **Stock Price Prediction**: Predict future stock prices using Linear Regression
- 📊 **Movement Classification**: Classify stock movements as up or down using Logistic Regression
- 🔍 **Investor Segmentation**: Segment investors into clusters using K-Means Clustering
- 📱 **Market Sentiment Analysis**: Analyze market sentiment (Coming Soon)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/smartfin-analyzer-plus.git
cd smartfin-analyzer-plus
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run smartfin_analyzer_plus.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Use the sidebar to navigate between different analysis modules:
   - Home: Overview and quick start guide
   - Stock Price Prediction: Predict future stock prices
   - Movement Classification: Classify stock movements
   - Investor Segmentation: Segment investors into clusters
   - Market Sentiment: Analyze market sentiment (Coming Soon)

## Data Sources

The application supports multiple data sources:
- CSV file upload
- Yahoo Finance API (yfinance)
- Finnhub API (requires API key)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the amazing web framework
- scikit-learn for machine learning capabilities
- Plotly for interactive visualizations
- yfinance and Finnhub for financial data 
