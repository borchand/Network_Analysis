import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import stockcorr as sc
from networkx.algorithms import community


def split_into_years(thresh = 0.9):
    ## IMPORTANT: Only works when index is not the date column
    stock_df = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
    stock_df['Date'] = pd.to_datetime(stock_df.index)
    first_date = stock_df['Date'].dt.year.drop_duplicates()

    ## split the stockdf into dataframes, one for each year

    dataframes_ = [stock_df[stock_df['Date'].dt.year == year] for year in first_date]
    dataframes_ = [df.set_index('Date') for df in dataframes_]
    corr_list = []
    for i, dataframe in enumerate(dataframes_):
        print(f"getting corr matrix {i} out of {len(dataframes_)}", end = "\r")
        curr_ = sc.get_corr_matrix(dataframe, thresh, verbose=False)
        ## make every value to 1 if above 0
        curr_ = np.where(curr_ > 0, 1, 0)

        corr_list.append(curr_)

    ## sum all correlation matrices in corr_list

    total_sum = np.sum(corr_list, axis=0)
    return corr_list, total_sum, dataframes_