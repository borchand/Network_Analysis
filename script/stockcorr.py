#stock correlation network
from re import I
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community.centrality import girvan_newman

start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2022, 1, 1)

#stock list
tickers = ('AAPL','TSLA', 'MSTF', 'AMZN', 'GOOG', 'FB', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'ADBE', 'CRM', 'INTC', 'CSCO', 'QCOM', 'TXN', 'AVGO', 'AMD','OXY')
#download stock data
df = web.DataReader(tickers, 'yahoo', start, end)
stockdf = df
#from csv
stockdf = pd.read_csv('outfile.csv', index_col=0)
print(stockdf.shape)
#count and remove nan
stockdf.isna().sum()
stockdf = stockdf.dropna(axis=1, how='all')
stockdf.shape



A = stockdf.corr().to_numpy()
A = np.where(abs(A) > .7, A, 0)
A = np.where(A == 1, 0, A)
A







G = nx.from_numpy_array(A)
G.remove_nodes_from(list(nx.isolates(G)))
widths = nx.get_edge_attributes(G, 'weight')
nodelist = G.nodes()

communities = girvan_newman(G)


node_groups = []

for com in next(communities):

  node_groups.append(list(com))

print(node_groups)
color_map = []

for node in G:
    if node in node_groups[0]:
        color_map.append('blue')
    else: 
        color_map.append('green')  
nx.draw(G, node_color=color_map, with_labels=True)
plt.show()