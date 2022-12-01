import networkx as nx
import numpy as np
import pandas as pd
from colour import Color
from matplotlib import animation
from matplotlib import pyplot as plt

from stockcorr import get_corr_matrix, relabel_graph

# Animate stocks so they go green if their price increases and red if it decreases
weeks = 52

# Load stock data csv
# Format is: stocks as columns and dates as rows
stockdf = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)

# Get subset of data in weeks
stockdf = stockdf.iloc[-weeks*5:]
# Get only one price per week
stockdf = stockdf.iloc[::5]


# Create subplot with network on top and small graph on bottom
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
# Make ax[1] smaller than ax[0]
ax[1].set_position([.1, .1, .8, .2])
# Make ax[0] take up the whole figure
ax[0].set_position([.1, .3, .8, .6])
# Remove the black border around the figure
ax[0].set_axis_off()
# Set border width on ax[1]
ax[1].spines['top'].set_linewidth(0.5)
ax[1].spines['right'].set_linewidth(0.5)
ax[1].spines['bottom'].set_linewidth(0.5)
ax[1].spines['left'].set_linewidth(0.5)
# Plot initial data on ax[1] (bottom subplot)
ax[1].plot(stockdf['COMP'], color='darkblue', linewidth=0.8)


threshold = 0.9

# Get correlation matrix
A = get_corr_matrix(stockdf, threshold=threshold)
# G = nx.read_gexf(f'./data/stockcorr_t{threshold}.gexf')
#standerd scaling
mask = A != 0
A[mask] = (A[mask] - A[mask].mean()) / A[mask].std()
G = nx.from_numpy_matrix(A, create_using=nx.Graph)
G = relabel_graph(G, stockdf.columns) # relabel nodes

# Get only first x nodes
num_nodes_to_keep = 100
G = G.subgraph(list(G.nodes)[:num_nodes_to_keep])

pos = nx.spring_layout(G)

# Draw edges
nx.draw_networkx_edges(G, pos=pos, alpha=.1, ax=ax[0])
# Draw labels
nx.draw_networkx_labels(G, pos=pos, font_size=7, ax=ax[0])

percentage_change_range = 11
red2gray = list(Color('red').range_to(Color('gray'), percentage_change_range))
gray2green = list(Color('gray').range_to(Color('green'), percentage_change_range-5))

def animate(week):
    # Set title at top of plot to week number
    fig.suptitle(f'Week {week+1} ({stockdf.index[week]})')
    # Get stock prices for this week
    stockprices = stockdf.iloc[week]
    # Get stock prices for first week
    last_stockprices = stockdf.iloc[0]

    # Calculate percentage change
    change = ((stockprices - last_stockprices) / last_stockprices) * 100

    mean_change = np.mean(change)
    change = change - mean_change

    # Color nodes based on percentage change
    colors = []
    for node in G.nodes:
        # Get percentage change
        pc = change[node]        
        # Get color
        if pc > 0:
            color = gray2green[min(percentage_change_range-1-5, int(pc))]
            colors.append(color.hex)
        elif pc < 0:
            color = red2gray[min(percentage_change_range-1, int(abs(pc)))]
            colors.append(color.hex)
        else:
            colors.append('#808080')

          
    # Get correlation matrix
    A = get_corr_matrix(stockdf.iloc[:week+1], threshold=threshold)
    # G = nx.read_gexf(f'./data/stockcorr_t{threshold}.gexf')
    #standerd scaling
    mask = A != 0
    A[mask] = (A[mask] - A[mask].mean()) / A[mask].std()
    new_G = nx.from_numpy_matrix(A, create_using=nx.Graph)
    # Get only first x nodes
    new_G = new_G.subgraph(list(new_G.nodes)[:num_nodes_to_keep])
    # Copy new_G to prevent frozen graph
    new_G = new_G.copy()
    new_G = relabel_graph(new_G, stockdf.columns) # relabel nodes  

    # Clear ax[0] (top subplot)
    ax[0].clear()

    # Draw the edges of the new graph
    nx.draw_networkx_edges(new_G, pos=pos, alpha=.1, ax=ax[0])    
    # Draw labels
    nx.draw_networkx_labels(new_G, pos=pos, font_size=7, ax=ax[0])
    # Draw nodes
    nx.draw_networkx_nodes(new_G, pos=pos, node_color=colors, node_size=20, ax=ax[0])

    # Plot current data on ax[1]
    ax[1].lines[0].set_data(stockdf['COMP'][:week+1].index, stockdf['COMP'][:week+1])
    # Plot dot on every week
    ax[1].scatter(stockdf['COMP'][:week+1].index, stockdf['COMP'][:week+1], s=1, color='darkblue')
    # Plot dot on future weeks
    ax[1].scatter(stockdf['COMP'][week+1:].index, stockdf['COMP'][week+1:], s=1, color='lightgray')

    # Plot COMP stock with green line if price increased and red if it decreased
    # ax[1].plot(stockdf['COMP'][0:week+1], color='g' if stockprices['COMP'] > last_stockprices['COMP'] else 'r')

    # Set y axis label
    ax[1].set_ylabel('Overall Market (USD)')
    # Tilt ax[1] x-labels
    plt.setp(ax[1].get_xticklabels(), rotation=90,)

    return fig


ani = animation.FuncAnimation(fig, animate, frames=weeks, interval=500, repeat=True)
plt.show()