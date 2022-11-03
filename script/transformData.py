#eda of full stock data.
import pandas as pd

def main():
    stockdf = pd.read_csv('../data/stock_market_data/all_data.csv')

    stockdf.drop(columns=['Unnamed: 0'], inplace=True)
    # make date column into datetime object
    stockdf['Date'] = pd.to_datetime(stockdf['Date'], dayfirst=True)
    #stockdf = stockdf[(stockdf['Date'] > '2010-1-1')]

    #pivot table to get the stock price for each company
    stockdf = stockdf.pivot_table(index='Date', columns='ticker', values='Close')

    #plot example stock prices
    stockdf[['AAPL','TWTR']].plot()

    #print to csv
    stockdf.to_csv('../data/stock_market_data/stockdf.csv')

if __name__ == "__main__":
    main()