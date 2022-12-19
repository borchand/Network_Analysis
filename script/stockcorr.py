#stock correlation network
import os
from sys import platform

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm

# set working directory to current directory
if platform == "darwin" or platform == "linux":
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    os.chdir(path)
else:
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    dir = os.path.dirname(path) #point to a directory
    os.chdir(dir)

def get_corr_matrix(df, threshold=0, from_file=False, verbose=True):
    """Calculate correlation given between column in dataframe.
    To include all columns, set threshold to 0, fx when constructing MST

    Args:
            df (DataFrame): Input dateframe
            threshold (float, optional): correlation threshold lower value will lead to a more connected graoh. Defaults to 0.9.

    Returns:
            ndarray: correlation matrix
    """
    if from_file == False:
        A = df.corr().to_numpy()
        if verbose:
            print('calculating corr matrix')
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
        A = df.corr().to_numpy()
        if verbose:
            print('calculating corr matrix')
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

def min_spanning_tree(A):
    dist = np.sqrt(2*(1-A)) # calc distance matrix
    mst = minimum_spanning_tree(dist) # get sparse mst matrix
    G = nx.from_scipy_sparse_matrix(mst) # convert to graph
    #count nan in dist
    return G

def graph_from_corr_matrix(A, threshold=0, from_file=False):
    if threshold == 0:
        G = nx.from_numpy_matrix(A, create_using=nx.Graph)
    elif threshold != 0:
        G = nx.from_numpy_matrix(A, create_using=nx.Graph, thresh = threshold)   
    return(G)

def split_into_years(threshold = 0.9, cor_edge_weight=False):
    ## IMPORTANT: Only works when index is not the date column
    stock_df = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
    stock_df['Date'] = pd.to_datetime(stock_df.index)
    first_date = stock_df['Date'].dt.year.drop_duplicates()

    ## split the stockdf into dataframes, one for each year

    dataframes_ = [stock_df[stock_df['Date'].dt.year == year] for year in first_date]
    dataframes_ = [df.set_index('Date') for df in dataframes_]
    corr_list = []
    for i, dataframe in enumerate(dataframes_):
        print(f"getting corr matrix {i} out of {len(dataframes_)}", end = "\r")
        curr_ = get_corr_matrix(df=dataframe, threshold=threshold, verbose=False)
        if cor_edge_weight:
            # make every value to 0 if below 0. And keep initial value if above 0
            curr_ = np.where(curr_ > 0, curr_ , 0)
        else:
            ## make every value to 1 if above 0
            curr_ = np.where(curr_ > 0, 1 , 0)

        corr_list.append(curr_)

    ## sum all correlation matrices in corr_list



    total_sum = np.sum(corr_list, axis=0)
    return corr_list, total_sum, dataframes_


def sorting_indexes(list_of_lists:list[list]):
    intersects = []

    for i in range(len(list_of_lists)):
        for j in range(i+1,len(list_of_lists)):
            set_i = set(list_of_lists[i])
            set_j = set(list_of_lists[j])
            intersect = set_i.intersection(set_j)
            intersects.append([list(intersect),i,j])

    idx_in_list = list(range(len(list_of_lists)))
    # print(idx_in_list)
    
    # sort intersect by length of the first element in the list
    intersects = sorted(intersects, key=lambda x: len(x[0]), reverse=True)
    # print(intersects)


    final_list = []
    stored_idx = None
    while idx_in_list != []:
        if stored_idx == None:
            if intersects[0][1] in idx_in_list:
                final_list.append(list_of_lists[intersects[0][1]])
                idx_in_list.remove(intersects[0][1])
            if intersects[0][2] in idx_in_list:
                final_list.append(list_of_lists[intersects[0][2]])
                idx_in_list.remove(intersects[0][2])
            stored_idx = intersects[0][2]
            intersects.pop(0)

        else:
            found = False
            for i in range(len(intersects)):
                if stored_idx == intersects[i][1]:
                    if intersects[i][2] in idx_in_list:
                        final_list.append(list_of_lists[intersects[i][2]])
                        idx_in_list.remove(intersects[i][2])
                    found = True
                    
                    intersects.pop(i)
                    break
                
                elif stored_idx == intersects[i][2]:
                    if intersects[i][1] in idx_in_list:
                        final_list.append(list_of_lists[intersects[i][1]])
                        idx_in_list.remove(intersects[i][1])
                    found = True

                    intersects.pop(i)
                    break
            if not found:
                stored_idx = None
                    
    # print(final_list)
    ## flatten final_list
    final_list = [item for sublist in final_list for item in sublist]

    final_list = list(dict.fromkeys(final_list))

    return final_list


def main():
    #get data
    thresh=0
    stockdf = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
    startlen = len(stockdf.columns)

    stockdf = stockdf.dropna(axis=1, how='all')
    print(f'dropped {startlen-len(stockdf.columns)} columns')
    
    #create mst from correlation matrix
    A = get_corr_matrix(stockdf, threshold=thresh)
    #standerd scaling
    mask = A != 0
    #A[mask] = (A[mask] - A[mask].mean()) / A[mask].std()
    #A

    print(A.shape) # get correlation matrix

    print('creating graph from correlation matrix')
    G = nx.from_numpy_matrix(A, threshold=thresh)

    G = relabel_graph(G, stockdf.columns) # relabel nodes

    # Save graph
    nx.write_gexf(G, f'../data/stockcorr_t{thresh}.gexf')


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
    main()
    


