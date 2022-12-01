import stockcorr as sc
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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
    subgraphs = {}

    all_nodes = list(corr_net)
    nodes_all = list(corr_net)

    while not len(all_nodes) == 0:
        node = all_nodes[0]
        bfs = nx.bfs_tree(corr_net, node)

        # add to dict
        subgraphs[node] = corr_net.subgraph(bfs)
        
        #remove from all_nodes
        for i in bfs: all_nodes.remove(i)

    edges_dict_subgraphs = {i: len(subgraphs[i].nodes()) for i in subgraphs}
    # get the key with the highest value item
    key = ""
    maxVal = 0

    for i in edges_dict_subgraphs:
        if edges_dict_subgraphs[i] > maxVal:
            maxVal = edges_dict_subgraphs[i]
            key = i

    main_tree = nx.Graph(subgraphs[key])
    subgraphs.pop(key)
    edges_dict_subgraphs.pop(key)
    for i in main_tree.nodes(): nodes_all.remove(i)

    while not len(nodes_all) == 0:
        max = 0

        for i in nodes_all:
            edges_ = connected_mst.edges(i)
            for j in edges_:
                if j[1] in main_tree.nodes() and connected_mst.get_edge_data(j[0], j[1])['weight'] > max:
                    max = connected_mst.get_edge_data(j[0], j[1])['weight']
                    node0 = i
                    node1 = j[1]
                    
        main_tree.add_edge(node0, node1, weight=max)

        try: 
            sub = subgraphs[node0]
            for y in sub.nodes():
                nodes_all.remove(y)
        except: 
            for y in subgraphs:
                if node0 in subgraphs[y].nodes():
                    for z in subgraphs[y].nodes():
                        nodes_all.remove(z)
    return main_tree

def main():
    stockdf = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
    stockdf = stockdf.dropna(axis=1, how='all')
    G = mst_and_relabel(stockdf)
    
    draw_mst(G)
    
    find_max_degree(G)
    G_without_max_degree = mst_and_relabel(stockdf.drop(columns=['BSAC']))
    draw_mst([G, G_without_max_degree])
    
    corr_net = create_corr_net(stockdf.drop(columns=['BSAC']), 0.9)
    nx.draw(corr_net, with_labels=False, node_size=10, alpha=.6, width=1)
    plt.show()

    connected_mst = G_without_max_degree
    main_tree = create_conected_net(connected_mst, corr_net)
    nx.draw(main_tree, with_labels=False, node_size=10, alpha=.6, width=1)
    
    plt.show()

if __name__ == '__main__':
    main()