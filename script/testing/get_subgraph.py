import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..") 
from networkx.algorithms import community
import sklearn.cluster as cluster

def get_list_clusters(clusters, corr_df):
    clusters_important = clusters.cluster_centers_indices_
    labels = clusters.labels_
    list_of_graphs = []

    sub_clusters = []
    for y in range(len(clusters_important)+1):
        list_of_nodes = []
        for i in np.argwhere(labels == y):
            list_of_nodes.append(i[0])
        sub_clusters.append(list_of_nodes)
        
    for i in range(len(clusters_important)):
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


def main():
    stockdf = pd.read_csv('./data/all_ticker_data.csv', index_col=0, parse_dates=True)
    year = 2020
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
    
    G = nx.from_numpy_matrix(corr_df)

    ## give the nodes a label
    # G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), stockdf.columns)))


    A = nx.to_scipy_sparse_array(G)


    clusters = cluster.AffinityPropagation().fit(A)
    
    get_list_clusters(clusters, corr_df)
    

if __name__ == '__main__':
    main()