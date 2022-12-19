import os
from sys import platform

import numpy as np
import pandas as pd
import networkx as nx
import sklearn.cluster as cluster
import pickle
from tqdm import tqdm

# set working directory to current directory
if platform == "darwin" or platform == "linux":
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    os.chdir(path)
else:
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    dir = os.path.dirname(path) #point to a directory
    os.chdir(dir)


def get_data(debug=False):
    if debug: print('Reading data...')
    
    df = pd.read_csv('../../data/all_ticker_data.csv', index_col=0, parse_dates=True)
    
    min_year = df.index.min().year
    max_year = df.index.max().year

    df_years = [df.loc[f'{year}-01-01':f'{year}-12-31'] for year in range(min_year, max_year + 1)]

    return df_years, min_year, max_year


def get_corr_from_year(year, df_years, min_year, to_numpy=True, debug=False):
    if debug: print('Spliting data in years...')
    
    year_data = df_years[year - min_year]

    # remove tickers that don't have any variation during the year

    # drop tickers that don't have data for the year
    year_data = year_data.dropna(axis=1, how='any')

    # convert price data to log returns
    log_returns = np.log(year_data).diff()[1:]
    
    return_variation = np.var(log_returns, axis=0).astype('float16')
    log_returns = log_returns[return_variation[return_variation > 0].index]

    # normalize log returns
    log_returns = (log_returns - log_returns.mean()) / log_returns.std()


    corr_df = log_returns.corr()

    return corr_df.to_numpy() if to_numpy else corr_df

def save_affinity_propagation_from_year(corr_df, year, debug=False):
    if debug: print('Create affinity propagation...')
    
    # create graph from correlation matrix
    G = nx.from_numpy_matrix(corr_df)
    
    # convert graph to scipy sparse matrix
    A = nx.to_scipy_sparse_array(G)
    
    # create affinity propagation from sparse matrix
    clusters = cluster.AffinityPropagation().fit(A)
    
    if debug: print('Save affinity propagation...')
    
    # open file where affinity propagation will be saved
    with open('../../Data/affinity_propagation_'+str(year), 'wb') as affinity_propagation_file:
        # save affinity propagation
        pickle.dump(clusters, affinity_propagation_file)
    
    return clusters

def read_affinity_propagation_from_year(year, debug=False):
    if debug: print('Reading affinity propagation...')
    # open file where affinity propagation will be read
    with open('../../Data/affinity_propagation_'+str(year), 'rb') as affinity_propagation_file:
        # get affinity propagation
        affinity_propagation = pickle.load(affinity_propagation_file)
 
    return affinity_propagation


def main():
    df_years, min_year, max_year = get_data(debug=True)

    corr_df = get_corr_from_year(2021, df_years, min_year, debug=True)
    
    print(corr_df.shape)
    for year in tqdm(range(min_year, max_year+1)):

        save_affinity_propagation_from_year(corr_df, year)
    
    affinity_propagation = read_affinity_propagation_from_year(2021, debug=True)
    


if __name__ == '__main__':
    main()