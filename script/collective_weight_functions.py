import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stockcorr as sc 
from networkx.algorithms import community

def split_into_years(threshold = 0.9, one_hot_where=False):
    ## IMPORTANT: Only works when index is not the date column
    stock_df = pd.read_csv('../data/stock_market_data/stockdf.csv')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    first_date = stock_df['Date'].dt.year.drop_duplicates()

    ## split the stockdf into dataframes, one for each year

    dataframes_ = [stock_df[stock_df['Date'].dt.year == year] for year in first_date]
    dataframes_ = [df.set_index('Date') for df in dataframes_]
    corr_list = []
    for dataframe in dataframes_:
        curr_ = sc.get_corr_matrix(df=dataframe, threshold=threshold)
        if one_hot_where:
            # make every value to 0 if below 0
            curr_ = np.where(curr_ > 0, curr_ , 0)
        else:
            ## make every value to 1 if above 0        
            curr_ = np.where(curr_ > 0, 1 , 0)
            
        corr_list.append(curr_)

    ## sum all correlation matrices in corr_list

    total_sum = np.sum(corr_list)
    return corr_list, total_sum, dataframes_