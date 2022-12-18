import sys
sys.path.append("..") 

from collective_weight_functions import split_into_years
import networkx as nx
import matplotlib.pyplot as plt


corr_, total_sum, dataframes_ = split_into_years(one_hot_where=True)


#nx matrix to graph
G = nx.from_numpy_matrix(total_sum)


#plot graph with small nodes

#make higher weight edges more visible
pos = nx.spring_layout(G, k=0.15, iterations=20)

weights = [e[2]["weight"] for e in G.edges.data()]
min_weight = min(weights)
max_weight = max(weights)
normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
# Set the edge colors based on the normalized weights
edge_colors = ["#%02x%02x%02x" % (min(255,int(300 * (1 - w))), min(255,int(300 * (1 - w))), min(255,int(300 * (1 - w)))) for w in normalized_weights]

plt.figure(figsize=(200,200))
nx.draw(G, pos, node_size=10)
plt.show()
