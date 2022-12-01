import datetime
import glob
import os
import numpy as np
import sys
from sys import platform
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

# set working directory to current directory
if platform == "darwin" or platform == "linux":
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    os.chdir(path)
else:
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    dir = os.path.dirname(path) #point to a directory
    os.chdir(dir)

def find_ETFs(datasets):
    """Find All non stocks from all json files

    Args:
        datasets (list): list of names of stock exchanges

    Returns:
        set: set containing all stocks
    """    
    x = set()
    print("find ETF's...")
    for dataset in datasets:
        print("Scaning %s" % dataset)
        all_files = glob.glob(f"../../data/stock_market_data/{dataset}/json/*.json")

        for file in tqdm(all_files):
            with open(file, 'r') as f:
                lines = [f.readline() for _ in range(10)]
                line = lines[8].split("\"")[3]
                symbol = lines[6].split("\"")[3]
                time = lines[9].split(":")[1].split(",")[0]
                if line != "EQUITY" or time == " null" or int(time) > 1420072583:
                    x.add(symbol)
    return x


def create_df(datasets, etfs):
    #Create empty dataframe with daterange index in range
    df = pd.DataFrame()
    df.index = pd.date_range(start='2005-1-1',end= '2023-1-1') 


    for dataset in datasets:
        print(f'reading JSON for {dataset}')
        # Get all files from /data/stock_market_data/dataset/csv
        all_files = glob.glob(f"../../data/stock_market_data/{dataset}/csv/*.csv")

        for file in tqdm(all_files):

            symbol = file.split("/")[-1].split(".")[0]
            
            #if not in etfs set merge to df
            if symbol not in etfs:
                #read csv file for corresponding symbol
                temp_df = pd.read_csv(file)

                #Set index of temp df to datetime and rename close column to symbol name (for merging)
                temp_df.index = pd.to_datetime(temp_df['Date'], format='%d-%m-%Y')
                temp_df.rename({'Close': symbol}, axis=1, inplace=True)

                #merge on index only from 2005
                df = df.merge(temp_df[symbol]['2005-1-1':], left_index=True, right_index=True, how='left')

    #remove days with 0 data ie. hollidays and weekends.
    df.dropna(axis=0, how='all', inplace=True)

    #drop duplicate columns by finding columns where pandas appended _x or _y to avoid conflict
    df.drop(list(df.filter(regex = '_')), axis = 1, inplace = True)
    return df

def adding_market_index(df):
    print("Add market index...")
    
    print("Read HistoricalData_1667914714805.csv...")
    ## Reading in the market_index and making sure that date is datetime so that easily mergable and that date is index
    hist_data = pd.read_csv('../../data/shared_data/HistoricalData_1667914714805.csv')
    hist_data['Date'] = pd.to_datetime(hist_data['Date'])
    hist_data.set_index('Date')

    ## Join df with hist_data on Date
    x = df.join(hist_data.set_index('Date')['Close/Last'], on='Date', how='left', lsuffix='_caller', rsuffix='_other').rename(columns={'Close/Last': 'COMP'})
    ## formating column and naming stock COMP
    return x

def drop_nan_std(stockdf):
    # 0 std check, and removing them from stockdf
    drop_list = []
    for i in stockdf.columns:
        if np.std(stockdf[i]) == 0:
            print(f'{i} has a standard deviation of {np.std(stockdf[i])}')
            drop_list.append(i)
    print(f'Dropping {len(drop_list)} columns')
    stockdf.drop(columns=drop_list, inplace=True)
    return stockdf

def sub_dataframe_to_csv(stockdf, start_date, end_date, fileName):
    ## create a mask where the date is between start and end
    print(f"Creating {fileName}.csv...")
    mask = (stockdf.index >= start_date) & (stockdf.index <= end_date)
    stockdf[mask].set_index(stockdf[mask].index).to_csv(f'../../data/stock_market_data/{fileName}.csv')
    

def clean_data(run_all=False):
    datasets = [
        "nyse",
        "nasdaq",
        "sp500",
        "forbes2000"
    ]
    if not run_all:
        try:
            print("Reading all_data.csv...")
            df = pd.read_csv('../../data/stock_market_data/stockdf.csv')
        except FileNotFoundError:
            run_all = True

    if run_all:
        etfs = find_ETFs(datasets)
        df = create_df(datasets, etfs)

    print(f"Saving to stockdf.csv...")
    #print to csv
    df.to_csv(f'../../data/stock_market_data/stockdf.csv')
    print(f"Total amount of tickers after cleaning: {len(df.columns)}")

if __name__ == '__main__':
    print(platform)
    if len(sys.argv) > 1:
        run_all = sys.argv[1].lower()[0] == "t"
    else:
        run_all = False
    clean_data(run_all)