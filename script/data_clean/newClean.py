import pandas as pd
import numpy as np
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
    
def get_data():
    df = pd.read_csv('../../data/all_ticker_data.csv', index_col=0, parse_dates=True)
    
    min_year = df.index.min().year
    max_year = df.index.max().year

    df_years = [df.loc[f'{year}-01-01':f'{year}-12-31'] for year in range(min_year, max_year + 1)]

    return df_years, min_year, max_year

def get_corr_from_year(year, df_years, min_year):
    year_data = df_years[year - min_year]

    # remove tickers that don't have any variation during the year
    year_variation = np.var(year_data, axis=0).astype('float16')
    year_data = year_data[year_variation[year_variation > 0].index]

    # drop tickers that don't have data for the year
    year_data = year_data.dropna(axis=1, how='any')
    
    corr_df = year_data.corr().to_numpy()
    
    return corr_df

def main():
    df_years, min_year, max_year = get_data()

    corr_df = get_corr_from_year(2020, df_years, min_year)
    
    print(corr_df.shape)

if __name__ == '__main__':
    main()