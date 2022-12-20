import os

import numpy as np
import pandas as pd
import yfinance as yf
from data_clean.newClean import get_data

MIN_YEAR = 2005
MAX_YEAR = 2021
SNP500_DF = pd.DataFrame()
SNP500_DF["snp500"] = yf.download('^GSPC', start=f'{MIN_YEAR}-01-01', end=f'{MAX_YEAR}-12-31', progress=False)["Adj Close"]


def baseline_backtest(year, pct_returns=False):
    """
    Backtest the S&P 500 on a given year

    Parameters
    ----------
    year : int
        year to backtest

    Returns
    -------
    float
        snp500 log returns for given year
    """
    if year < MIN_YEAR or year > MAX_YEAR:
        raise ValueError(f"year must be between {MIN_YEAR} and {MAX_YEAR}")

    year_data = SNP500_DF.loc[f'{year}-01-01':f'{year}-12-31']
    year_data = year_data.fillna(method='ffill')

    if pct_returns:
        snp_returns = year_data.pct_change()[1:]
        snp_returns = snp_returns.fillna(0)
        snp_returns = (snp_returns + 1).cumprod()
        return snp_returns["snp500"] - 1

    snp_returns = (np.log(year_data) - np.log(year_data.shift(1)))[1:]
    snp_returns = snp_returns.fillna(0)
    snp_returns = snp_returns.cumsum()

    return snp_returns["snp500"]


def backtest(year, data, tickers, pct_returns=False):
    """
    Backtest the S&P 500 on a given year

    Parameters
    ----------
    year : int
        year to backtest
    
    data : list
        list of dataframes with adj close data for all tickers, one for each year
    
    tickers : list
        list of tickers to backtest on

    pct_returns : bool
        whether to return pct returns or log returns. Default is False

    Returns
    -------
    float
        snp500 returns for given year
    """
    if year < MIN_YEAR or year > MAX_YEAR:
        raise ValueError(f"year must be between {MIN_YEAR} and {MAX_YEAR}")

    year_data = data[year - MIN_YEAR]

    # get the data for the tickers
    year_data = year_data[tickers]

    # drop tickers that don't have data for the year
    year_data = year_data.dropna(axis=1, how='any')
    
    # year returns
    if pct_returns:
        pct_returns = year_data.pct_change()[1:]
        pct_returns = pct_returns.fillna(0)
        pct_returns = (pct_returns + 1).cumprod()
        pct_returns = (pct_returns - 1).mean(axis=1)
        return pct_returns

    log_returns = (np.log(year_data) - np.log(year_data.shift(1)))[1:]
    log_returns = log_returns.fillna(0)
    log_returns = log_returns.cumsum()
    log_returns = log_returns.mean(axis=1)

    return log_returns



def main():
    # Testing
    baseline_backtest(2021, pct_returns=True).plot()
    baseline_backtest(2021).plot()

    # Testing backtesting
    year = 2021
    data, _, _ = get_data()
    tickers = ["AAPL"]
    backtest(year, data, tickers, pct_returns=True).plot()


if __name__ == '__main__':
    main()