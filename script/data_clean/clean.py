import pandas as pd
import glob
from tqdm import tqdm

def find_ETFs(datasets):
    x = set()
    print("find ETF's...")
    for dataset in datasets:
        print("Scaning %s" % dataset)
        all_files = glob.glob(f"./data/stock_market_data/{dataset}/json/*.json")

        for file in tqdm(all_files):
            with open(file, 'r') as f:
                lines = [f.readline() for _ in range(10)]
                line = lines[8].split("\"")[3]
                symbol = lines[6].split("\"")[3]
                time = lines[9].split(":")[1].split(",")[0]

                if line != "EQUITY" or time == " null" or int(time) > 1420072583:
                    x.add(symbol)
    return x


def merge_data_to_csv(datasets, etfs):

    dataframes = []

    for dataset in datasets:
        print("Downloading %s" % dataset + "...")
        # Get all files from /data/stock_market_data/dataset/csv
        all_files = glob.glob(f"./data/stock_market_data/{dataset}/csv/*.csv")

        # For each file in all_files, read the csv file and append it to a list
        df_list = []
        for file in tqdm(all_files):
            etf = file.split("/")[-1].split(".")[0]
            if not etf in etfs:
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

    print("Concatenating all...")
    # Load data
    all_df = pd.concat([dataframe for dataframe in dataframes])

    print("Deleting duplicate tickers...")
    # Remove duplicate rows
    df = all_df.drop_duplicates()

    print("Remove delisted...")
    nan_tickers = df[df['ticker'].isna()]['ticker'].unique()

    df = df[~df['ticker'].isin(nan_tickers)]

    print("Printing to all_data.csv...")
    # print df to csv
    df.to_csv('./data/stock_market_data/all_data.csv')
    return df

def transform_data(stockdf):
    print("Transform dataframe...")
    if "Unnamed: 0" in stockdf.columns:
        stockdf.drop(columns=['Unnamed: 0'], inplace=True)
    # make date column into datetime object
    stockdf['Date'] = pd.to_datetime(stockdf['Date'], dayfirst=True)

    #pivot table to get the stock price for each company
    stockdf = stockdf.pivot_table(index='Date', columns='ticker', values='Close')

    return stockdf



def adding_market_index(df):
    print("Add market index...")
    
    print("Read HistoricalData_1667914714805.csv...")
    ## Reading in the market_index and making sure that date is datetime so that easily mergable and that date is index
    hist_data = pd.read_csv('./data/shared_data/HistoricalData_1667914714805.csv')
    hist_data['Date'] = pd.to_datetime(hist_data['Date'])
    hist_data.set_index('Date')

    ## Join df with hist_data on Date
    x = df.join(hist_data.set_index('Date')['Close/Last'], on='Date', how='left', lsuffix='_caller', rsuffix='_other').rename(columns={'Close/Last': 'COMP'})
    ## formating column and naming stock COMP
    return x

def save_to_csv(stockdf):
    print("Print to stockdf.csv...")
    #print to csv
    stockdf.to_csv('./data/stock_market_data/stockdf.csv')

def clean_data(run_all=False):
    datasets = [
        "nyse",
        "nasdaq",
        "sp500",
        "forbes2000"
    ]
    if run_all:
        etfs = find_ETFs(datasets)
        df = merge_data_to_csv(datasets, etfs)
    else:
        print("Reading all_data.csv...")
        df = pd.read_csv('./data/stock_market_data/all_data.csv')

    stockdf = transform_data(df)
    stockdf = adding_market_index(stockdf)
    save_to_csv(stockdf)

if __name__ == '__main__':
    clean_data()