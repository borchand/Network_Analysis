import pickle
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt

sys.path.append('../data_clean/')
import newClean as nc

def get_indecies_of_labels(year, df_years):
    affinity_prop = nc.read_affinity_propagation_from_year(year)
    clusters_list = nc.read_yearly_clusters(year)
    for i in range(len(df_years)):
        if df_years[i].index.min().year == year:
            stock_df = df_years[i]
    tickers_in_cluster = []
    for i in range(affinity_prop.cluster_centers_indices_.shape[0]):
        tickers_in_cluster.append(np.argwhere(affinity_prop.labels_ == i))
        
    return tickers_in_cluster, clusters_list,affinity_prop, stock_df

def closeness_centrality(clusters_list, stock_df):
    tickers = []
    for G in clusters_list:
        # set weight in graph to abs
        abs_values = [abs(x[2]['weight']) for x in G.edges(data=True)]
        for i,j in enumerate(G.edges(data=True)):
            j[2]['weight'] = abs_values[i]
        
        bc = nx.closeness_centrality(G, distance='weight')
        keyValue = [(v, k) for k, v in bc.items()]
        
        # sort keyValue on values
        keyValue.sort(key = lambda x: x[0])

        maxKey = keyValue[-1][1]
        
        ticker = stock_df.columns[maxKey]
        tickers.append(ticker)

    return tickers
    
def current_flow_betweenness_centrality(clusters_list, stock_df):

    tickers = []
    for G in clusters_list:
        # set weight in graph to abs
        abs_values = [abs(x[2]['weight']) for x in G.edges(data=True)]
        for i,j in enumerate(G.edges(data=True)):
            j[2]['weight'] = abs_values[i]
        
        bc = nx.current_flow_betweenness_centrality(G, weight='weight')
        keyValue = [(v, k) for k, v in bc.items()]
        
        # sort keyValue on values
        keyValue.sort(key = lambda x: x[0])
        
        maxKey = keyValue[-1][1]
        
        ticker = stock_df.columns[maxKey]
        tickers.append(ticker)

    return tickers
    
    
def main():
    df_years, min_year, max_year = nc.get_data()
    tickers_in_cluster, clusters_list, affinity_prop, stock_df = get_indecies_of_labels(2021, df_years)


    print(closeness_centrality(clusters_list, stock_df))
    print(current_flow_betweenness_centrality(clusters_list, stock_df))

if __name__ == '__main__':
    main()