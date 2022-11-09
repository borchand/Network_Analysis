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

def get_corr_matrix(df, threshold=0.9):
    """Calculate correlation given between column in dataframe.
    To include all columns, set threshold to 0, fx when constructing MST

    Args:
            df (DataFrame): Input dateframe
            threshold (float, optional): correlation threshold lower value will lead to a more connected graoh. Defaults to 0.9.

    Returns:
            ndarray: correlation matrix
    """
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

thresh = .985
def main(thresh):
    #get data

    stockdf = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
    startlen = len(stockdf.columns)
    stockdf = stockdf.dropna(axis=1, how='all')
    print(f'dropped {startlen-len(stockdf.columns)} columns')

    #create mst from correlation matrix
    A = get_corr_matrix(stockdf, threshold=thresh) # get correlation matrix
    if thresh == 0:
        print('creating mst from correlation matrix')
        dist = np.sqrt(2*(1-A)) # calc distance matrix
        mst = minimum_spanning_tree(dist) # get sparse mst matrix
        G = nx.from_scipy_sparse_matrix(mst) # convert to graph
    else:
        print('creating graph from correlation matrix')
        G = nx.from_numpy_matrix(A, create_using=nx.Graph) # convert to graph

    G = relabel_graph(G, stockdf.columns) # relabel nodes

    #count nan in dist
    print(f'number of nan in dist: {np.isnan(dist).sum()}')

    #get list of degrees sorted
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    deg
    #draw graphs
    tickerlst = 'AAPL,AMZN,GOOG,MSFT'.split(',')
    fig, ax = plt.subplots(1,4, figsize=(20,10))
    for i, ticker in enumerate(tickerlst):
        ax[i].set_title(ticker)
        sub = get_neighborhood(G, ticker, 2)
        H = G.subgraph(sub)
        nx.draw(H, with_labels=True, ax=ax[i])
    plt.show()

    # remove singletons
    G.remove_nodes_from(list(nx.isolates(G)))
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
    pos = nx.spring_layout(G)
    colorlist = [ 'r', 'g', 'b', 'c', 'm', 'y', 'brown', 'orange', 'purple' ]
    communities = girvan_newman(G)
    setLst = list(next(communities))
    len(setLst)
    plt.figure(figsize = (50,30))
    for index, sg in enumerate(setLst):
        #draw with orange color text
        nx.draw(G.subgraph(sg), pos= pos, node_color= colorlist[index % 7], with_labels=True)
    plt.show()

if __name__ == "__main__":
    main(thresh)
    
