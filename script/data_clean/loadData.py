import datetime
import glob
import os
import numpy as np
import sys
from sys import platform
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

if platform == "darwin":
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
        all_files = glob.glob(f"../data/stock_market_data/{dataset}/json/*.json")
        for file in tqdm(all_files):
            with open(file, 'r') as f:
                lines = [f.readline() for _ in range(10)]
                line = lines[8].split("\"")[3]
                symbol = lines[6].split("\"")[3]
                time = lines[9].split(":")[1].split(",")[0]
                if line != "EQUITY" or time == " null" or int(time) > 1420072583:
                    x.add(symbol)
    return x

def clean_data(run_all=False):
    datasets = [
        "nyse",
        "nasdaq",
        "forbes2000",
        "sp500"
    ] 
    #Dont look at composite index'es

    if not run_all:
        try:
            print("Reading all_data.csv...")
            df = pd.read_csv('../../data/stock_market_data/Stockdf.csv')
        except FileNotFoundError:
            run_all = True

    if run_all:
        etfs = find_ETFs(datasets)

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

df = clean_data(run_all = True)

df.to_csv('Stockdf.csv', index=True)