#stock correlation network
from operator import index
from sqlite3 import Row
from threading import Thread
import pandas_datareader.data as web
import datetime
import pandas as pd
from time import sleep, time
from tqdm import tqdm


start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2022, 1, 1)
date = pd.date_range(start, end)
date = pd.DataFrame(date)
inputDf = pd.read_csv('data/tickers.csv')
tickers = inputDf[f'Symbol'].tolist()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

lst = []
def getstocks(tickers):
    #download stock data
    s = time()
    stocks = web.get_data_yahoo(tickers, start, end)['Close']
    stocks = pd.DataFrame(stocks)
    
    ## merge the stocks and date on date
    
    stocks = pd.merge(date, stocks, how='left', left_on=0, right_on='Date').rename(columns={'Close':tickers})
    
    #print(stocks)
    #date = date.join(date, on = 'Date', as_index = f'{tickers}', lsuffix = '_caller', rsuffix = '_other')
    return stocks


threads = []
for tickerchunk in tqdm(range(len(tickers))):
    sleep(1)
    try:
        date = getstocks(tickers[tickerchunk])
    except:
        pass


date.to_csv('ClosefileBig.csv',index=True, header=True, chunksize=100)


