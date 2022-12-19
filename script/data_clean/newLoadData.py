import yfinance as yf
import os
from sys import platform

# set working directory to current directory
if platform == "darwin" or platform == "linux":
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    os.chdir(path)
else:
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    dir = os.path.dirname(path) #point to a directory
    os.chdir(dir)
    
def create_ticker_txt():
    # should be added later (not important for now)
    return

def main():
    # not tested
    with open("../data/tickers.txt", "r") as f:
        tickers = [ticker.strip() for ticker in f.readlines()]

    df = yf.download(tickers, start='2005-01-01', end='2021-12-31')['Adj Close']
    
    df.to_csv('../../data/all_ticker_data.csv')
    
if __name__ == '__main__':
    main()