#stock correlation network
from re import I
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community.centrality import girvan_newman
from scipy.sparse.csgraph import minimum_spanning_tree
import os 
from networkx.algorithms import community

def get_corr_matrix(df, threshold=0.9, from_file=False):
    """Calculate correlation given between column in dataframe.
    To include all columns, set threshold to 0, fx when constructing MST

    Args:
            df (DataFrame): Input dateframe
            threshold (float, optional): correlation threshold lower value will lead to a more connected graoh. Defaults to 0.9.

    Returns:
            ndarray: correlation matrix
    """
    if from_file == False:
        print('calculating corr matrix')
        A = df.corr().to_numpy()
        print(f'A has {np.isnan(A).sum()} nan values')

        if threshold != 0:
            A = np.where(abs(A) > threshold, A, 0)

        A = np.where(A == 1, 0, A)
        return A

    threshString = str(threshold).lstrip('0.')

    if os.path.isfile(f'../data/corr_matrix_t{threshString}.npy'):
        print('loading corr matrix from file')
        return np.load(f'../data/corr_matrix_t{threshString}.npy')

    else:
        print('calculating corr matrix')
        A = df.corr().to_numpy()
        print(f'A has {np.isnan(A).sum()} nan values')

        if threshold != 0:
            A = np.where(abs(A) > threshold, A, 0)

        A = np.where(A == 1, 0, A)
        with open(f'../data/corr_matrix_t{threshString}.npy', 'wb') as f:
            np.save(f, A)
            
            return A

def relabel_graph(G, labels):
    """Relabels graph nodes with labels

    Args:
            G (graph): input graph
            labels (label list): list of labels

    Returns:
            graph: returns graph with relabeled nodes
    """
    nodes = {i:labels[i] for i in range(len(labels))}
    nx.relabel_nodes(G, nodes, copy=False)
    return G


def get_neighborhood(G, node, depth=1):
    """Creates recursively a neighborhood from starting node

    Args:
            G (Graph): input graph
            node (node): start node
            depth (int, optional): Depth at which to include nodes. Defaults to 1.

    Returns:
            set: returns set of nodes in neighborhood including starting node 
    """
    if depth == 0:
            return {node}
    else:
            return {node}.union(*[get_neighborhood(G, neighbor, depth-1) for neighbor in G.neighbors(node)])

thresh = .9
def main(thresh):
    #get data

    stockdf = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
    startlen = len(stockdf.columns)
    stockdf = stockdf.dropna(axis=1, how='all')
    print(f'dropped {startlen-len(stockdf.columns)} columns')
    
    #create mst from correlation matrix
    A = get_corr_matrix(stockdf, threshold=thresh)
    #standerd scaling
    mask = A != 0
    A[mask] = (A[mask] - A[mask].mean()) / A[mask].std()
    A

    print(A.shape) # get correlation matrix
    if thresh == 0:
        print('creating mst from correlation matrix')
        dist = np.sqrt(2*(1-A)) # calc distance matrix
        mst = minimum_spanning_tree(dist) # get sparse mst matrix
        G = nx.from_scipy_sparse_matrix(mst) # convert to graph
    else:
        print('creating graph from correlation matrix')
        G = nx.from_numpy_matrix(A, create_using=nx.Graph) # convert to graph

    G = relabel_graph(G, stockdf.columns) # relabel nodes


    #get list of degrees sorted
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    deg
    #draw graphs
    tickerlst = 'AAPL,AMZN,GOOG,MSFT'.split(',')
    fig, ax = plt.subplots(1,4, figsize=(80,20))
    for i, ticker in enumerate(tickerlst):
        ax[i].set_title(ticker)
        sub = get_neighborhood(G, ticker, 1)
        H = G.subgraph(sub)
        nx.draw(H, with_labels=True, ax=ax[i], alpha=.6, node_size=1000, font_size=20, width=1)
    plt.show()

    # Count number of nodes in whole graph
    print(f'Number of nodes: {len(G.nodes)}')
    # remove singletons
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f'Number of nodes after removing singletons: {len(G.nodes)}')
    print(f'That is {len(G.nodes)/len(stockdf.columns)*100:.2f}% of all stocks')
    
    #draw network with colored connected components
    pos = nx.spring_layout(G)
    colorlist = [ 'r', 'g', 'b', 'c', 'm', 'y', 'brown', 'orange', 'purple' ]
    wcc = nx.connected_components( G )
    setLst = list(wcc) 
    len(setLst)
    plt.figure(figsize = (50,30))
    for index, sg in enumerate(setLst):
        #draw with orange color text
        nx.draw(G.subgraph(sg), pos= pos, node_color= colorlist[index % 7], with_labels=True, font_size=20)
    plt.show()

    #draw network with colored communities
    communities = community.greedy_modularity_communities(G)
    #position by community
    colorlist = [ 'r', 'g', 'b', 'c', 'm', 'y', 'brown', 'orange', 'purple' ]
    
    setLst = list(communities)
    plt.figure(figsize = (50,30))
    for index, sg in enumerate(setLst):
        #draw with orange color text
        print(index)
        pos = nx.spring_layout(G.subgraph(sg))
        nx.draw(G.subgraph(sg), pos= pos, node_color= colorlist[index % 7], with_labels=True)
    plt.show()

if __name__ == "__main__":
    main(thresh)
    


