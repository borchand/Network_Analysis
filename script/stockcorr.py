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


def get_corr_matrix(df, threshold=0.9):
  """Calculate correlation given between column in dataframe

  Args:
      df (DataFrame): Input dateframe
      threshold (float, optional): correlation threshold lower value will lead to a more connected graoh. Defaults to 0.9.

  Returns:
      ndarray: correlation matrix
  """
  A = df.corr().to_numpy()
  A = np.where(abs(A) > .9, A, 0)
  A = np.where(A == 1, 0, A)
  return A

def relabel_graph(G, labels):
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

def main():
  #get data
  stockdf = pd.read_csv('../data/AllfileBig.csv', index_col=0)
  startlen = len(stockdf.columns)
  stockdf = stockdf.dropna(axis=1, how='all')
  print(f'dropped {startlen-len(stockdf.columns)} columns')

  #create mst from correlation matrix
  A = get_corr_matrix(stockdf, threshold=0) # get correlation matrix
  dist = np.sqrt(2*(1-A)) # calc distance matrix
  mst = minimum_spanning_tree(dist) # get sparse mst matrix
  Gmst = nx.from_scipy_sparse_matrix(mst) # convert to graph

  #relabel graph
  Gmst = relabel_graph(Gmst, stockdf.columns)


  Gmst.remove_node('DOUG') # higly correlated with alot of stocks
  
  #draw graphs
  tickerlst = 'AAPL,AMZN,GOOG,MSFT'.split(',')
  fig, ax = plt.subplots(1,4, figsize=(20,10))
  for i, ticker in enumerate(tickerlst):
    ax[i].set_title(ticker)
    sub = get_neighborhood(Gmst, ticker, 4)
    H = Gmst.subgraph(sub)
    nx.draw(H, with_labels=True, ax=ax[i])
  plt.show()
  
 

if __name__ == "__main__":
  main()
  
