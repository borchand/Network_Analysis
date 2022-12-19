import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib as mpl

import networkx as nx
import pandas as pd
import numpy as np
import sys
sys.path.append(".") 
import stockcorr as sc 
from networkx.algorithms import community
import sklearn.cluster as cluster

def load_data_for_year(year, stockdf):
    # get the data for the year
    min_year = stockdf.index.min().year
    max_year = stockdf.index.max().year
    years = [stockdf.loc[f'{year}-01-01':f'{year}-12-31'] for year in range(min_year, max_year + 1)]
    year_data = years[year - min_year]

    year_variation = np.var(year_data, axis=0)
    year_data = year_data[year_variation[year_variation > 0].index]

    # drop tickers that don't have data for the year
    year_data = year_data.dropna(axis=1, how='any')

    corr_df = year_data.corr().to_numpy()
    return corr_df

def initialise_clusters(df):
    G = nx.from_numpy_matrix(corr_df)

    A = nx.to_scipy_sparse_array(G)

    clusters = cluster.AffinityPropagation().fit(A)
    
    return clusters

def create_box_data(df,clusters):
    all_data = []
    n_labels = clusters.labels_.max()
    for y in range(n_labels+1):
        data = []
        for i in np.argwhere(clusters.labels_ == y):
            data.append(df[i, y][0])
        all_data.append((y,data))

    all_data = sorted(all_data, key=lambda x: np.median(x[1]))
    original_index = [x[0] for x in all_data]
    all_data = [x[1] for x in all_data]
    bp = plt.boxplot(all_data, patch_artist=True)
    for box in bp["boxes"]:
        box.set( facecolor='gray')
    for whisker in bp["whiskers"]:
        whisker.set(color='orange', linewidth=2)
    for cap in bp["caps"]:
        cap.set(color='black', linewidth=2)
    for median in bp["medians"]:
        median.set(color='white', linewidth=2)
    for flier in bp["fliers"]:
        flier.set(marker='o', color='red')
    
    return bp, original_index
        





def main():
    stockdf = pd.read_csv('../data/all_ticker_data.csv', index_col=0, parse_dates=True)
    years = [2010,2015,2020,2021]   
    corr_dfs = []
    for year in years:
        corr_dfs.append(load_data_for_year(year, stockdf))
    fig, ax= plt.subplots(len(years),1, figsize=(20,10))
    for i in range(years):
        ax[i] = fig.add_axes([0,0,1,1])
        cluster_ = initialise_clusters(corr_dfs[i])
        bp, indexes = create_box_data(corr_dfs[i], cluster_)
        ax[i].set_xticklabels(indexes)
        
    plt.show()

        
    
    
    return

if __name__ == '__main__':
    main()