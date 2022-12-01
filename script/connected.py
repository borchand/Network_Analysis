import stockcorr as sc
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 

def mst_and_relabel(stockdf):
    A = sc.get_corr_matrix(stockdf, threshold=0, from_file=False)
    mst = sc.min_spanning_tree(A)
    G = sc.relabel_graph(mst, stockdf.columns)
    return G

def draw_mst(G: nx.Graph or list(nx.Graph)):
    if type(G) != list:
        nx.draw(G, with_labels=False, node_size=10, alpha=0.6, width=1)
    else:
        l = len(G)
        fig, ax = plt.subplots(1,l, figsize=(80,20))
        for i in range(l):
            nx.draw(G[i], with_labels=False, node_size=10, alpha=0.6, width=1, ax=ax[i])
    plt.show()

def find_max_degree(G : nx.Graph):
    edges_dict = {i: len(G[i]) for i in G.nodes()}
    ## get the key with the highest value item
    key = max(edges_dict, key=edges_dict.get)
    return key, len(G[key])

def create_corr_net(stockdf: pd.DataFrame, thresh=0.9):
    A = sc.get_corr_matrix(stockdf, thresh, from_file=False)
    corr_net = nx.from_numpy_matrix(A, create_using=nx.Graph)
    corr_net = sc.relabel_graph(corr_net, stockdf.columns)
    return corr_net

def create_conected_net(connected_mst: nx.Graph, corr_net: nx.Graph):
    corr_net = corr_net.copy()
    connected_mst = connected_mst.copy()
    largest_comp = max(nx.connected_components(corr_net), key=len)
    main_tree = nx.subgraph(corr_net, largest_comp).copy()

    for i in connected_mst.edges():
        main_tree.add_edge(i[0], i[1])
  
    return main_tree

def main():
    stockdf = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
    G = mst_and_relabel(stockdf)
    
    draw_mst(G)
    
    # find_max_degree(G)
    # G_without_max_degree = mst_and_relabel(stockdf.drop(columns=['BSAC']))
    # draw_mst([G, G_without_max_degree])
    
    # corr_net = create_corr_net(stockdf, 0.9)
    # nx.draw(corr_net, with_labels=False, node_size=10, alpha=.6, width=1)
    # plt.show()

    nx.is_connected(main_tree)
    len(list(nx.connected_components(main_tree)))

    main_tree = create_conected_net(G, corr_net)
    nx.draw(main_tree, with_labels=False, node_size=10, alpha=.6, width=1)
    
    # plt.show()

if __name__ == '__main__':
    main()