#stock correlation network
from re import I
import datetime
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community.centrality import girvan_newman
from scipy.sparse.csgraph import minimum_spanning_tree
import os 
from networkx.algorithms import community
from matplotlib.animation import FuncAnimation
from colour import Color

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


stockdf = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
startlen = len(stockdf.columns)
stockdf = stockdf.dropna(axis=1, how='all')
print(f'dropped {startlen-len(stockdf.columns)} columns')

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

#get percentage change of stock prices weekly
percentStock = stockdf.pct_change(7)
#get mean of weeks
percentStock.index = pd.to_datetime(percentStock.index)
#percentStock = percentStock.groupby(percentStock.index.week).mean()
# pos = nx.spring_layout(G)
red = Color("red")
green = Color("green")
grey = Color("grey")
#get first row
percentStock.iloc[8]
redTgrey = list(red.range_to(grey, 20))
greyTgreen = list(grey.range_to(green, 20))
def get_color(val):
    if val < 0:
        try: 
            return redTgrey[min(int(abs(val)*20),20)].hex
        except:
            return grey.hex
    else:
        try:
            return greyTgreen[min(int(val*20),20)].hex
        except:
            return grey.hex

fig = plt.figure(figsize=(30,10))
G = G.subgraph(get_neighborhood(G, 'AAPL', depth=1))
plt.title('stock correlation network', fontsize=50)
nx.draw_networkx_edges(G, pos= pos, alpha=0.1)
nx.draw_networkx_labels(G, pos= pos, font_size=20)

def animate(day):

    nx.draw_networkx_nodes(G, pos= pos, node_color=[get_color(percentStock.iloc[day][node]) for node in G.nodes])

    plt.title(f'week: {day}', fontsize=50)
    plt.tight_layout() 

anim = FuncAnimation(fig, animate, frames=range(0, len(percentStock)))
plt.show()


fig = plt.figure(figsize=(30,10))
plt.plot(percentStock.index, percentStock['AAPL'])
#plot apple through time
def animate(day):

    # plt.plot(percentStock.index, percentStock['AAPL'].array()[:day], color='black')
    plt.title(f'week: {day}', fontsize=50)
    plt.tight_layout()

anim = FuncAnimation(fig, animate, frames=range(0, len(percentStock)))
plt.show()