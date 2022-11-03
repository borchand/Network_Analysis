import pandas as pd
import glob
from tqdm import tqdm
import json 

def merge_data_to_csv(datasets):
    dataframes = []

    for dataset in datasets:
        print("Downloading %s" % dataset + "...")
        # Get all files from /data/stock_market_data/dataset/csv
        all_files = glob.glob(f"../../data/stock_market_data/{dataset}/csv/*.csv")

        # For each file in all_files, read the csv file and append it to a list
        df_list = []
        for file in tqdm(all_files):
            df = pd.read_csv(file)
            df_list.append(df)

        # Create a dataframe with all the data, including a column with the file name
        print("Concatenating %s" % dataset + "...")
        df = pd.concat(df_list, keys=all_files)

        # Extract the ticker from the file name. The ticker is right before .csv
        df['ticker'] = df.index.get_level_values(0).str.split('/').str[-1].str.split('.').str[0]
        
        # Reset the index
        df = df.reset_index(drop=True)
        # Add the dataframe to a list
        dataframes.append(df)

    print("Deleting duplicate tickers...")
    # Load data
    all_df = pd.concat([dataframe for dataframe in dataframes])
    # Remove duplicate rows
    df = all_df.drop_duplicates()
    # Download all_data
    print("Saving to all_data.csv...")
    df.to_csv('../../data/stock_market_data/all_data.csv', index=False)

def remove_ETFs(datasets):
    x = set()
    for dataset in datasets:
        print("Scaning %s" % dataset)
        all_files = glob.glob(f"../../data/stock_market_data/{dataset}/json/*.json")

        for file in tqdm(all_files):
            with open(file, 'r') as f:
                lines = [f.readline() for _ in range(9)]
                line = lines[8].split("\"")[3]
                symbol = lines[6].split("\"")[3]

                if line in ['ETF', 'MUTUALFUND']:
                    x.add(symbol)

    print("Total: %s" % len(x))

    df = pd.read_csv('../../data/stock_market_data/all_data.csv')
    print("Length before removing ETFs: %s" % len(df))
    df = df[~df['ticker'].isin(x)]
    print("Length after removing ETFs: %s" % len(df))

    df.to_csv('../../data/stock_market_data/equity_data.csv', index=False)
    print("Data saved to equity_data.csv")


def main():
    datasets = [
        "nyse",
        "nasdaq",
        "sp500",
        "forbes2000"
    ]
    merge_data_to_csv(datasets)
    remove_ETFs(datasets)

if __name__ == '__main__':
    main()