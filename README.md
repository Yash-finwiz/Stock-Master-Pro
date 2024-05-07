
# Stock Master Pro


Welcome to the repository for StockMaster Pro, a sophisticated web application designed for stock market forecasting and trading strategy formulation. This document provides an overview of the project, including its design, features, and usage instructions.
## Deployment

To deploy this project run

**STEP 1:** Clone repository.
  ```bash
git clone https://github.com/YASH260/Stock-Master-Pro.git

```

  **STEP 2:** Change the directory to the repository.
```bash
  cd stock-master-pro
```

**STEP 3:** Create a virtual environment
(For Windows)
```bash
  python -m venv virtualenv
```
(For MacOS and Linux)
```bash
  python3 -m venv virtualenv
```

**STEP 4:** Activate the virtual environment.
(For Windows)
```bash
  virtualenv\Scripts\activate
```
(For MacOS and Linux)
```bash
  source virtualenv/bin/activate
```

**STEP 5:** Install the dependencies.
```bash
  pip install -r requirements.txt
```

**STEP 6:** Migrate the Django project.
(For Windows)
```bash
  python manage.py migrate
```
(For MacOS and Linux)
```bash
  python3 manage.py migrate
```

**STEP 7:** Run the application.
(For Windows)
```bash
  python manage.py runserver
```
(For MacOS and Linux)
```bash
  python3 manage.py runserver
```





## Project Overview

StockMaster Pro leverages advanced machine learning models and technical analysis tools to offer predictions and actionable insights for stock market trading. The application integrates several state-of-the-art technologies and strategies


## Stock Master Pro is comprised of 5 main modules:
1. Price Prediction
2. Sentimental Analysis
3. Pattern Analyzer
4. Algorithmic Trading
5. Technical Scan

##


## 1. Stock Price Forecasting
StockMaster Pro incorporates three powerful forecasting models, each selected for their robustness and effectiveness in handling different aspects of stock price prediction. Here's a more detailed look at each:


## a. using Lstm

Utilized for more complex data sequences where the order and timing of historical data points matter. LSTM captures long-term dependencies and can remember information over extended periods, making it suitable for predicting stock price movements over longer horizons.

## b. using Arima

Primarily employed for short-term forecasting of non-seasonal time series data. ARIMA analyzes trends and non-constant variance to predict future stock prices based on past price movements.

## c. using Prophet
Designed to handle time series data with strong seasonal effects and historical trend changes. Prophet accommodates missing data and outliers, making it robust for forecasting stocks affected by anomalies or external events.


## 2. Sentimental Analysis
This module incorporates market sentiment analysis as a key component for predicting market movements based on public perception. This analysis involves the utilization of XLNet for Reddit and RoBERTa for Twitter to extract valuable insights from textual data and gauge the sentiment of market participants.
## a. XLNet for Reddit:
XLNet, a powerful language model, is utilized to analyze textual data from Reddit. By applying XLNet, StockMaster Pro can extract sentiment-related insights from Reddit posts and comments, providing an understanding of the overall sentiment towards specific stocks or market trends.

## b. RoBERTa for Twitter:
StockMaster Pro leverages RoBERTa, an advanced language model, to analyze textual data from Twitter. By utilizing RoBERTa's natural language processing capabilities, StockMaster Pro can extract sentiment-related information from tweets, enabling an understanding of the sentiment of the market based on public opinions and discussions.

## 3. Pattern Analyzer
StockMaster Pro utilizes TA-Lib for automatic candlestick pattern recognition. This powerful technical analysis library identifies and analyzes candlestick patterns in stock price charts, providing valuable insights for traders and investors. By leveraging TA-Lib, StockMaster Pro helps users identify potential market trends and make informed decisions based on historical price patterns.

``` bash
Inputs:
1. Stock name
2. Pattern name
Output:
1. A visually appealing plot with requested pattern.
```


## 4. Algorithmic Trading
For any requested stock, this module now backtests a hardcoded trading strategy and generates a visually appealing report with information on the number of trades, total returns, maximum drawdown, and average return.
``` bash
Inputs:
1. Stock name

Output:
1. A plot indicating the backtest results for the requested stock.  
```


## 5. Technial Scan
StockMaster Pro integrates a KNN-based algorithm for real-time market trend analysis and buy/sell signal identification. This algorithm considers Rate of Change (ROC), Commodity Channel Index (CCI), Volume, and Relative Strength Index (RSI) as features. By analyzing these indicators, StockMaster Pro provides users with concise and precise insights into market trends and potential trading opportunities

``` bash
Inputs:
1. Stock name

Output:
1. A visually appealing plot indicating buy and sell signals for the requested stock.

```


## Contributing

* Please feel free to suggest improvements, bugs by creating an issue.
* Please follow the [Guidelines for Contributing](Stock-Master-Pro\CONTRIBUTING.md) while making a pull request.
* Stay tuned for Upcoming Features for a list of exciting features we plan to implement in the future.

## Disclaimer
* DO NOT use the results provided by the web app 'solely' to make your trading/investment decisions.
* Always backtest and analyze the stocks manually before you trade.
* Consult your financial advisor before making any trading/investment decisions.
* The authors/contributors and the web app will not be held liable for your losses (if any).