import os
import pickle
from sys import platform

import networkx as nx
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
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
    import hashlib
    with open('../../data/all_ticker_data.csv', 'rb') as f:
        data = f.read()
        sha = hashlib.sha256(data).hexdigest()
    
    if sha == '7a852906a443e0ca99e9654290457674fd4dfc3d1a795313c92c2bf8cb3975d1':
        df = pd.read_csv('../../data/all_ticker_data.csv', index_col=0, parse_dates=True)
    
    else:
        print('Data is corrupted, please download again')
        return
    
    min_year = df.index.min().year
    max_year = df.index.max().year

    df_years = [df.loc[f'{year}-01-01':f'{year}-12-31'] for year in range(min_year, max_year + 1)]

    return df_years, min_year, max_year


def get_list_clusters(clusters, corr_df):    
    clusters_important = clusters.cluster_centers_indices_
    labels = clusters.labels_
    list_of_graphs = []

    sub_clusters = []
    sub_corr = []
    for y in range(len(clusters_important)+1):
        list_of_nodes = []
        list_of_corr = []
        for i in np.argwhere(labels == y):
            list_of_corr.append(corr_df[i, y][0])
            list_of_nodes.append(i[0])
        sub_clusters.append(list_of_nodes)
        sub_corr.append(list_of_corr)

        
    for i in tqdm(range(len(clusters_important)),leave=False):
        main_node = clusters_important[i]
        G = nx.empty_graph()
        G.add_node(main_node)
        for y in range(len(sub_clusters[i])):
            if main_node != sub_clusters[i][y]:
                G.add_edge(main_node, sub_clusters[i][y], weight=sub_corr[i][y])
        for y in sub_clusters[i]:
            for z in sub_clusters[i]:
                if z != y:
                    G.add_edge(z,y, weight=corr_df[y,z])
        list_of_graphs.append(G)
    return list_of_graphs


def get_corr_from_year(year, df_years, min_year, to_numpy=True, debug=False):
    if debug: print('Spliting data in years...')
    
    # get data from year
    year_data = df_years[year - min_year]

    # convert price data to log returns
    log_returns = np.log(year_data).diff()[1:]

    # normalize log returns
    log_returns = (log_returns - log_returns.mean()) / log_returns.std()

    # drop nan log returns
    log_returns = log_returns.dropna(axis=1, how='any')

    # get correlation matrix
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
    with open('../../Data/Affinity_propagation/affinity_propagation_'+str(year), 'wb') as affinity_propagation_file:
        # save affinity propagation
        pickle.dump(clusters, affinity_propagation_file)
    
    return clusters


def save_clusters_years_list(corr_df, year, clusters,debug=False):
    if debug: print('Save affinity propagation...')
    x = get_list_clusters(clusters, corr_df)
    with open(f'../../data/clustered_graphs/clustered_{year}', 'wb') as f:
        pickle.dump(x, f)


def read_affinity_propagation_from_year(year, debug=False):
    if debug: print('Reading affinity propagation...')
    # open file where affinity propagation will be read
    with open('../../Data/Affinity_propagation/affinity_propagation_'+str(year), 'rb') as affinity_propagation_file:
        # get affinity propagation
        affinity_propagation = pickle.load(affinity_propagation_file)
 
    return affinity_propagation


def read_yearly_clusters(year, debug=False):
    if debug: print('Reading affinity propagation...')
    # open file where affinity propagation will be read
    with open(f'../../data/clustered_graphs/clustered_{year}', 'rb') as cluster_file:
        # get affinity propagation
        cluster_years = pickle.load(cluster_file)
 
    return cluster_years


def main():
    df_years, min_year, max_year = get_data(debug=True)

    for year in tqdm(range(min_year, max_year+1)):

        corr_df = get_corr_from_year(year, df_years, min_year)
        clusters = save_affinity_propagation_from_year(corr_df, year)
        save_clusters_years_list(corr_df, year, clusters)

if __name__ == '__main__':
    main()