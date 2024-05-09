# Stock Master Pro


Welcome to the repository for StockMaster Pro, a sophisticated Django-based web application designed for stock market forecasting and trading strategy formulation. This document provides an overview of the project, including its design, features, and usage instructions.
## Deployment

To deploy this project run

**STEP 1:** Clone repository.
  ```bash
git clone https://github.com/Yash-finwiz/Stock-Master-Pro.git

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

![Screenshot 2024-05-05 205514](https://github.com/YASH260/Stock-Master-Pro/assets/59645048/91a54fbf-cd44-4c4a-958d-9aec2ba2466b)

##


## 1. Stock Price Forecasting
StockMaster Pro incorporates three powerful forecasting models, each selected for its robustness and effectiveness in handling different aspects of stock price prediction. Here's a more detailed look at each:


## a. using Lstm

Utilized for more complex data sequences where the order and timing of historical data points matter. LSTM captures long-term dependencies and can remember information over extended periods, making it suitable for predicting stock price movements over longer horizons.

![Screenshot 2024-05-07 203003](https://github.com/YASH260/Stock-Master-Pro/assets/59645048/38301186-a6aa-430b-83ab-e493c55be582)


## b. using Arima

Primarily employed for short-term forecasting of non-seasonal time series data. ARIMA analyzes trends and non-constant variance to predict future stock prices based on past price movements.

![Screenshot 2024-04-21 205406](https://github.com/YASH260/Stock-Master-Pro/assets/59645048/43be638a-0abf-4336-93e1-09dae081812e)

## c. using Prophet
Designed to handle time series data with strong seasonal effects and historical trend changes. Prophet accommodates missing data and outliers, making it robust for forecasting stocks affected by anomalies or external events.

![Screenshot 2024-04-21 205552](https://github.com/YASH260/Stock-Master-Pro/assets/59645048/ff4986ab-bc1f-484a-97c9-a2b2c156ea49)


## 2. Sentimental Analysis
This module incorporates market sentiment analysis as a key component for predicting market movements based on public perception. This analysis involves the utilization of XLNet for Reddit and RoBERTa for Twitter to extract valuable insights from textual data and gauge the sentiment of market participants.


## a. XLNet for Reddit:
XLNet, a powerful language model, is utilized to analyze textual data from Reddit. By applying XLNet, StockMaster Pro can extract sentiment-related insights from Reddit posts and comments, providing an understanding of the overall sentiment towards specific stocks or market trends.

## b. RoBERTa for Twitter:
StockMaster Pro leverages RoBERTa, an advanced language model, to analyze textual data from Twitter. By utilizing RoBERTa's natural language processing capabilities, StockMaster Pro can extract sentiment-related information from tweets, enabling an understanding of the sentiment of the market based on public opinions and discussions.
``` bash
Inputs:
1. Stock name
Output:
1. A pie chart 
```
![Screenshot 2024-04-21 210302](https://github.com/YASH260/Stock-Master-Pro/assets/59645048/06b24e61-163f-44bc-b579-695ff25a4d5f)

## 3. Pattern Analyzer
StockMaster Pro utilizes TA-Lib for automatic candlestick pattern recognition. This powerful technical analysis library identifies and analyzes candlestick patterns in stock price charts, providing valuable insights for traders and investors. By leveraging TA-Lib, StockMaster Pro helps users identify potential market trends and make informed decisions based on historical price patterns.

``` bash
Inputs:
1. Stock name
2. Pattern name
Output:
1. A visually appealing plot with the requested pattern.
```
![Screenshot 2024-05-07 203250](https://github.com/YASH260/Stock-Master-Pro/assets/59645048/5e1d19e5-7d5a-4ce5-91a4-ee3177590037)


## 4. Algorithmic Trading
For any requested stock, this module now backtests a hardcoded trading strategy. It generates a visually appealing report with information on the number of trades, total returns, maximum drawdown, and average return.
``` bash
Inputs:
1. Stock name

Output:
1. A plot indicating the backtest results for the requested stock.  
```
![trade_plot](https://github.com/YASH260/Stock-Master-Pro/assets/59645048/15f6873c-efbc-4a15-8df7-8d8c7861aba0)


## 5. Technical Scan
StockMaster Pro integrates a KNN-based algorithm for real-time market trend analysis and buy/sell signal identification. This algorithm considers Rate of Change (ROC), Commodity Channel Index (CCI), Volume, and Relative Strength Index (RSI) as features. By analyzing these indicators, StockMaster Pro provides users with concise and precise insights into market trends and potential trading opportunities

``` bash
Inputs:
1. Stock name

Output:
1. A visually appealing plot indicates the requested stock's buy and sell signals.

```
![Screenshot 2024-04-21 211100](https://github.com/YASH260/Stock-Master-Pro/assets/59645048/5a8ccf8e-7ac0-49f7-9a22-5eebdf5aaba0)


## Contributing

* Please feel free to suggest improvements, or bugs by creating an issue.
* Please follow the [Guidelines for Contributing](https://github.com/YASH260/Stock-Master-Pro/blob/main/CONTRIBUTING.md) while making a pull request.
* Stay tuned for Upcoming Features for a list of exciting features we plan to implement in the future.

## Disclaimer
* DO NOT use the results provided by the web app 'solely' to make your trading/investment decisions.
* Always backtest and analyze the stocks manually before you trade.
* Consult your financial advisor before making any trading/investment decisions.
* The authors/contributors and the web app will not be held liable for your losses (if any).
