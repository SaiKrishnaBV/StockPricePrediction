# StockPricePrediction

## File: main.ipynb

## Goal:
1. Comparative analysis of various predictive models to determine their effectiveness in estimating stock's closing prices.
2. Stock Price Prediction using Sentiment Analysis.
3. Evaluate the adaptability of trained models by predicting the stock price of a company not previously encountered.

## Dataset: 
1. Stocknet Dataset: Utilizing the Stocknet dataset for stock price prediction with sentiment analysis.
2. Custom Dataset: Prepared using yfinance containing the historical data of the companies.

## Baselines
    AutoArima
    Prophet

## Models:
1. Regression Models
		Linear Regression
		DecisionTreeRegressor
		AdaBoostRegressor
		RandomForestRegressor
		Support vector regressor
		XGBoostRegressor
2. LSTM

3. Sentiment Analysis and price prediction using XGBoostRegressor

## Experimentation:
1. Using a model trained on a company to predict the stock price of an unseen company by fine tuning

2. Giskard

3. Stock price prediction using Technical Indicators

## File: utility.py
Utility functions to load the dataset, plot the predictions, etc.