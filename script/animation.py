import networkx as nx
from matplotlib import pyplot as plt, animation
import random
import pandas as pd
import numpy as np
from colour import Color


# Animate stocks so they go green if their price increases and red if it decreases
weeks = 52

# Load stock data csv
# Format is: stocks as columns and dates as rows
stockdf = pd.read_csv('./data/stock_market_data/stockdf.csv', index_col=0)

# Get subset of data in weeks
stockdf = stockdf.iloc[-weeks*5:]
# Get only one price per week
stockdf = stockdf.iloc[::5]


# Create subplot with network on top and small graph on bottom
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

G = nx.read_gexf('./data/stockcorr_t0.9.gexf')
# Get only first 10 nodes
G = G.subgraph(list(G.nodes)[:100])
pos = nx.spring_layout(G)

# Draw edges
nx.draw_networkx_edges(G, pos=pos, alpha=.1, ax=ax[0])
# Draw labels
nx.draw_networkx_labels(G, pos=pos, font_size=7, ax=ax[0])

percentage_change_range = 11
red2gray = list(Color('red').range_to(Color('gray'), percentage_change_range))
gray2green = list(Color('gray').range_to(Color('green'), percentage_change_range-5))

def animate(week):
    # fig.clear()
    # Set title to week number
    plt.title(f'Week {week+1}')
    # Get stock prices for this week
    stockprices = stockdf.iloc[week]
    # Get stock prices for last week
    if week > 0:
        last_stockprices = stockdf.iloc[week-1]
    else:
        last_stockprices = stockprices

    # Calculate mean change in all stocks

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

            
        # Draw node
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=[node], node_color=color, node_size=1000, alpha=.8)

    # Color nodes based on price change
    # colors = []
    # for node in G.nodes:
    #     if stockprices[node] > last_stockprices[node]:
    #         colors.append('g')
    #     else:
    #         colors.append('r')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=20, ax=ax[0])

    # PLot a line of mean_change
    ax[1].plot(stockdf.index, stockdf['COMP'].to_numpy()[:week], color='black')
    # ax[1].plot(week, mean_change, 'o', color='black')

    # nx.draw(G, with_labels=True, pos=pos, node_color=colors)

    # Return figure
    return fig


ani = animation.FuncAnimation(fig, animate, frames=weeks, interval=500, repeat=True)
plt.show()