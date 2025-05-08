# Financial ML Analytics

An enhanced financial machine learning tool built with Streamlit.

## Features

- 📈 **Stock Price Prediction**: Predict future stock prices using Linear Regression
- 📊 **Movement Classification**: Classify stock movements as up or down using Logistic Regression
- 🔍 **Investor Segmentation**: Segment investors into clusters using K-Means Clustering
- 📱 **Market Sentiment Analysis**: Analyze market sentiment (Coming Soon)
- **Improved Feature Selection:** Target column cannot be selected as a feature, and the app warns if any feature is identical to the target.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/financial-ml-analytics.git
cd financial-ml-analytics
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

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501 or http://localhost:8504)

3. Use the sidebar to navigate between different analysis modules:
   - Home: Overview and quick start guide
   - Stock Price Prediction: Predict future stock prices
   - Movement Classification: Classify stock movements
   - Investor Segmentation: Segment investors into clusters
   - Market Sentiment: Analyze market sentiment (Coming Soon)

4. **Feature Selection:** The target column cannot be selected as a feature. If you select a feature that is identical to the target, a warning will appear. Remove such features for realistic evaluation.

## Data Sources

The application supports multiple data sources:
- CSV file upload
- Yahoo Finance API (yfinance)
- Finnhub API (requires API key)

## Troubleshooting

- **Yahoo Finance Download Errors:**
  - If you see errors like `JSONDecodeError('Expecting value: line 1 column 1 (char 0)')`, it may be due to network issues, symbol unavailability, or Yahoo Finance API rate limits. Try again after a few minutes, check your internet connection, or use a different symbol.
- **Model Evaluation Shows Perfect Scores (MAE=0, RMSE=0, R²=1):**
  - This usually means your features are identical to the target or your data is too small/synthetic. Use realistic data and ensure your features are not the same as the target.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the amazing web framework
- scikit-learn for machine learning capabilities
- Plotly for interactive visualizations
- yfinance and Finnhub for financial data 
