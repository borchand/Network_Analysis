import os
import pickle
import sys
# set working directory to current directory
from sys import platform

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

if platform == "darwin" or platform == "linux":
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    os.chdir(path)
else:
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    dir = os.path.dirname(path) #point to a directory
    os.chdir(dir)

sys.path.append('../data_clean/')
import newClean as nc

sys.path.append('../')
import backtesting as bt


def get_indecies_of_labels(year, df_years, abs=False):
    affinity_prop = nc.read_affinity_propagation_from_year(year, abs=abs)
    clusters_list = nc.read_yearly_clusters(year, abs=abs)
    for i in range(len(df_years)):
        if df_years[i].index.min().year == year:
            stock_df = df_years[i]
    tickers_in_cluster = []
    for i in range(affinity_prop.cluster_centers_indices_.shape[0]):
        tickers_in_cluster.append(np.argwhere(affinity_prop.labels_ == i))
        
    return tickers_in_cluster, clusters_list,affinity_prop, stock_df

def page_rank_top_node(tickers_in_cluster,clusters_list, affinity_prop, stock_df):
    most_important_tickers_plural = []
    for i in range(affinity_prop.cluster_centers_indices_.shape[0]):
        current_cluster = clusters_list[i]
        abs_values = [abs(x[2]['weight']) for x in current_cluster.edges(data=True)]
        for i,j in enumerate(current_cluster.edges(data=True)):
            j[2]['weight'] = abs_values[i]
        pagerank = nx.pagerank(current_cluster)
        sorted_ = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_)
        top_dog = sorted_[0][0]
        ticker = stock_df.columns[top_dog]
        most_important_tickers_plural.append(ticker)
    return most_important_tickers_plural

def eigenvector_centrality_top_node(tickers_in_cluster,clusters_list, affinity_prop, stock_df):
    most_important_tickers_plural = []
    for i in range(affinity_prop.cluster_centers_indices_.shape[0]):
        current_cluster = clusters_list[i]
        abs_values = [abs(x[2]['weight']) for x in current_cluster.edges(data=True)]
        for i,j in enumerate(current_cluster.edges(data=True)):
            j[2]['weight'] = abs_values[i]
        pagerank = nx.eigenvector_centrality(current_cluster)
        sorted_ = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        top_dog = sorted_[0][0]
        ticker = stock_df.columns[top_dog]
        most_important_tickers_plural.append(ticker)
    return most_important_tickers_plural

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

def centrality_all(year, df_years, min_year, abs=False):
    second_year = year+1
        
    tickers_in_cluster, clusters_list, affinity_prop, stock_df = get_indecies_of_labels(year, df_years, abs=abs)
    
    closeness = closeness_centrality(clusters_list, stock_df)
    current_flow = current_flow_betweenness_centrality(clusters_list, stock_df) 
    page_rank = page_rank_top_node(tickers_in_cluster,clusters_list, affinity_prop, stock_df)
    eigen_vector = eigenvector_centrality_top_node(tickers_in_cluster,clusters_list, affinity_prop, stock_df)
    
    baseline = bt.baseline_backtest(second_year, df_years, pct_returns=True)
    
    # closeness = bt.backtest(second_year, df_years, closeness, pct_returns=True)
    
    # current_flow = bt.backtest(second_year, df_years, current_flow, pct_returns=True)
    
    page_rank =bt.backtest(second_year, df_years, page_rank, pct_returns=True)
    
    # eigen_vector =bt.backtest(second_year, df_years, eigen_vector, pct_returns=True)
    
    affinity_prop_cluster_indices = df_years[year-min_year].columns[affinity_prop.cluster_centers_indices_]
    affinity_prop = bt.backtest(second_year, df_years, affinity_prop_cluster_indices, pct_returns=True)
    
    
    
    return baseline, page_rank, affinity_prop
    
def plot_centrality_backtest(year, df_years, min_year, abs=False):
    
    baseline, page_rank, affinity_prop = centrality_all(year, df_years, min_year, abs=abs)

    sns.set_style('darkgrid')    
    
    plt.plot(baseline, label='Baseline', color='black', linewidth=2)
    
    # plt.plot(closeness, label='Closeness', color='red', linewidth=2)
    
    # plt.plot(current_flow, label='Current Flow', color='blue', linewidth=2)
    
    # set all values that exceed a 10% change from previous day to nan
    # page_rank = page_rank[page_rank.pct_change() > 0.1]

    plt.plot(page_rank, label='Page Rank', color='orange', linewidth=2)
    
    # plt.plot(eigen_vector, label='Eigen Vector', color='orange', linewidth=2)
    
    plt.plot(affinity_prop, label='Affinity Propagation', color='blue', linewidth=2)
    # set legend fontsize
    plt.legend(fontsize=24)
    plt.title(f'Backtest of centrality measures from {year} compared to {year+1}', fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()
    
    
def main():
    df_years, min_year, max_year = nc.get_data()
    plot_centrality_backtest(2019, df_years, min_year)
    
    
        
    

if __name__ == '__main__':
    main()