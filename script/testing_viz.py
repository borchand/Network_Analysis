import itertools

import collective_weight_functions as cw
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from cdlib import algorithms


def main():
    corr_list, total_sum, dataframes_ = cw.split_into_years(threshold=0.95, one_hot_where=True)
    
    G = nx.from_numpy_matrix(corr_list[0], create_using=nx.Graph)

    # get the number of nodes and edges
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    print(f'Number of nodes: {nodes}')
    print(f'Number of edges: {edges}')

    # get the largest connected component as a graph
    print(nx.is_connected(G))
    Gc = max(nx.connected_components(G), key=len)
    G = nx.subgraph(G, Gc)
    print(nx.is_connected(G))

    # get the number of nodes and edges
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    print(f'Number of nodes: {nodes}')
    print(f'Number of edges: {edges}')

    # relabel nodes from 0 to n-1
    G = nx.convert_node_labels_to_integers(G, first_label=0)

    # community detection
    communities = algorithms.core_expansion(G, tolerance=0.001)
    communities_list = communities.communities
    print(len(communities_list))

    com_idx_sort = cw.soting_indexes(communities_list)
    len(com_idx_sort)

    # sort nodes by community
    com_idx = list(itertools.chain.from_iterable(communities_list))

    # remove duplicates after first occurence but keep order
    com_idx_sort = []
    count = 0
    for i in com_idx:
        if i not in com_idx_sort:
            com_idx_sort.append(i)
        else:
            count += 1
    print(count)

    for i in range(nodes):
        if i not in com_idx_sort:
            com_idx_sort.append(i)

    len(com_idx_sort)

    # get adjacency matrix of G
    adj = nx.to_numpy_matrix(G)
    print(adj.shape)

    # heatmap of adjacency matrix
    adj_mod = adj[com_idx_sort, :]
    adj_mod = adj_mod[:, com_idx_sort]
    plt.figure(figsize=(10, 10))
    plt.imshow(adj_mod, cmap='hot', interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main()