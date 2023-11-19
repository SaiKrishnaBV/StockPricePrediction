import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
import yfinance as yf
import pandas as pd
from GoogleNews import GoogleNews
import matplotlib.pyplot as plt

from selenium import webdriver 
from selenium.webdriver.chrome.service import Service as ChromeService 
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from giskard import Model, Dataset, scan, testing, GiskardClient, demo, Suite



def get_best_sliding_window(train_data, test_data, scaler, window_min_size=5, window_max_size=100):

    results = []
    for window_size in range(window_min_size,window_max_size):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        sliding_window = window_size
        for i in range(sliding_window, len(train_data)):
            x_train.append(train_data[i-sliding_window:i])
            y_train.append(train_data[i])

        for i in range(sliding_window, len(test_data)):
            x_test.append(test_data[i-sliding_window:i])
            y_test.append(test_data[i])

        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=16, epochs=5)
        predictions = model.predict(x_test)
        predictions_inv = scaler.inverse_transform(predictions)

        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
        print("Window size: {} Root mean square error: {}".format(window_size, rmse))

        results.append([window_size, rmse]) 

    min_window = 9999
    min_rmse = 999999999
    for elem in results:
        if(elem[1] < min_rmse):
            min_window = elem[0]
            min_rmse = elem[1]

    return min_window


def load_US_Stock_Data():

    tickers_us = ['AAPL', 'ABB', 'ABBV', 'AEP', 'AGFS', 'AMGN', 'AMZN', 'BA', 'BABA', 'BAC', 'BBL', 'BCH', 'BHP', 'BP', 'BRK-A', 'BSAC', 'BUD', 'C', 'CAT', 'CELG',
                'CHL', 'CHTR', 'CMCSA', 'CODI', 'CSCO', 'CVX', 'D', 'DHR', 'DIS', 'DUK', 'EXC', 'FB', 'GD', 'GE', 'GMRE', 'GOOG', 'HD', 'HON', 'HRG', 'HSBC', 'IEP', 
                'INTC', 'JNJ', 'JPM', 'KO', 'LMT', 'MA', 'MCD', 'MDT', 'MMM', 'MO', 'MRK', 'MSFT', 'NEE', 'NGG', 'NVS', 'ORCL', 'PCG', 'PCLN', 'PEP', 'PFE', 'PG', 'PICO',
                'PM', 'PPL', 'PTR', 'RDS-B', 'REX', 'SLB', 'SNP', 'SNY', 'SO', 'SPLP', 'SRE', 'T', 'TM', 'TOT', 'TSM', 'UL', 'UN', 'UNH', 'UPS', 'UTX', 'V', 'VZ', 'WFC', 'WMT', 'XOM']
    
    if not os.path.exists('data/price_us'):
        os.makedirs('data/price_us')

    for symbol in tickers_us:
        stock = yf.Ticker(symbol)
        data = stock.history(period="max")
        if not data.empty:
            file_path = os.path.join('data/price_us', f'{symbol}.csv')
            data.to_csv(file_path)

def StockEdge_scrape_data():
    url     = 'https://web.stockedge.com/share/infosys/4926?section=news' 
    company = 'Infosys'
    options = webdriver.ChromeOptions() 
    
    # run browser in headless mode 
    options.headless = True 

    driver = webdriver.Chrome(service=ChromeService( 
        ChromeDriverManager().install()), options=options) 

    driver.get(url) 
    timeout = 5
    try:
        element_present = EC.presence_of_element_located((By.TAG_NAME, 'se-content'))
        WebDriverWait(driver, timeout).until(element_present)
    except TimeoutException:
        print("Timed out waiting for page to load")

    elements = driver.find_elements(By.CLASS_NAME, 'full-height-required') 

    news = []
    date = []

    for title in elements: 
        try:
            items = title.find_elements(By.TAG_NAME, 'ion-item')
            print(len(items))
            for item in items:
                item_label = item.find_element(By.TAG_NAME, 'ion-label')

                if (item_label.get_attribute('innerText')):
                    news.append(item_label.get_attribute('innerText'))
        except:
            pass

        try:
            dates = title.find_elements(By.TAG_NAME, 'ion-item-divider')

            print(len(dates))
            for news_date in dates:
                news_label = news_date.find_element(By.TAG_NAME, 'ion-label')
                news_date_label = news_label.find_element(By.TAG_NAME, 'se-date-label')
                if(news_date_label.get_attribute('innerText')):
                    date.append(news_date_label.get_attribute('innerText'))
        except:
            pass

    output = list(zip(date, news))
    df = pd.DataFrame(output, columns = ['Date', 'News'])
    df.to_csv(company + ".csv")

def get_news():
    start_date     = '01/01/2020'
    end_date       = '08/31/2023'
    search_keyword = 'Infosys'

    googlenews=GoogleNews(start=start_date,end=end_date)
    googlenews.search(search_keyword)

    result=googlenews.result()
    df=pd.DataFrame(result)

    for i in range(2,10):
        googlenews.getpage(i)
        result=googlenews.result()
        print(googlenews.total_count())
        print("Page number: {} number of results: {}".format(i,len(result)))
        df=pd.DataFrame(result)

        googlenews.clear()

    req_news = pd.DataFrame(df[['datetime', 'title']])
    req_news.to_csv("news_" + search_keyword + ".csv")


# Function to plot actual and predicted values
def plot_actual_predicted_val(train, test, preds):

    valid = test.copy(deep=True)
    valid['Predictions'] = preds

    idx = list(range(0, train.shape[0] + test.shape[0]))
    train.index = idx[:train.shape[0]]
    valid.index = idx[train.shape[0]:]

    plt.plot(train['Close'], 'green')
    plt.plot(valid[['Close', 'Predictions']], label=["Actual", "Predicted"])
    plt.ylabel("Stock price")
    plt.legend()

# Giskard wrapper for model
class MyCustomModel(Model):
    def model_predict(self, df):
        return np.squeeze(self.model.predict(df))

# Function to get giskard analysis for the model
def get_giskard_analysis(x_test, y_test, target, model, model_name):
    inp_data = pd.concat([x_test, y_test], axis=1)

    giskard_dataset = Dataset(
        df=inp_data,
        target=target,
        name=model_name,
    )

    giskard_model = MyCustomModel(
        model=model,
        model_type="regression",
        feature_names=x_test.columns
    )

    y_pred = giskard_model.predict(giskard_dataset).raw_prediction

    # Calculating RMSE
    mse = mean_squared_error(y_test, y_pred)
    print("Test RMSE: ", np.sqrt(mse))

    scan_results = scan(giskard_model, giskard_dataset)
    display(scan_results)