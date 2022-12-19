import sys
import networkx as nx
import numpy as np
import pandas as pd
from colour import Color
from matplotlib import animation
from matplotlib import pyplot as plt

sys.path.append('./script')

import stockcorr as sc

# from stockcorr import relabel_graph

# Animate stocks so they go green if their price increases and red if it decreases
years = 18
# Load stock data csv
# Format is: stocks as columns and dates as rows
stockdf = pd.read_csv('../data/all_ticker_data.csv')

corr_, total_sum, dataframes_ = sc.split_into_years()

nasdaq_price = pd.read_csv('../data/nasdaq_index.csv')
# # Turn year_df indexes into datetime
nasdaq_price.index = pd.to_datetime(nasdaq_price['Date'])
# Get first row in each year from dataframes_combined
dataframes_combined_yearly = nasdaq_price.groupby(nasdaq_price.index.year).first()

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
## Get the average % change in price for each date
# avg_change = dataframes_[0].pct_change().mean(axis=0)
ax[1].plot(dataframes_combined_yearly['Close'], color='darkblue', linewidth=0.8)
# Make x ticks only display full year
ax[1].xaxis.set_major_locator(plt.MaxNLocator(10))

threshold = 0.9

# Get correlation matrix
A = corr_[0]
# G = nx.read_gexf(f'./data/stockcorr_t{threshold}.gexf')
G = nx.from_numpy_matrix(A, create_using=nx.Graph)
G = sc.relabel_graph(G, dataframes_[0].columns) # relabel nodes

# Check if all stocks_to_include are in G
cluster_centers_indices = np.array([  16,   26,  115,  127,  192,  224,  248,  270,  310,  312,  326,
        371,  527,  536,  593,  673,  723,  748,  759,  776, 1029, 1044,
       1058, 1120, 1125, 1184, 1259, 1323, 1370, 1387, 1438, 1476, 1580,
       1597, 1628, 1696, 1760, 1765, 1783, 1792, 1805, 1880, 1958, 2041,
       2066, 2218, 2238, 2330, 2339, 2439, 2463, 2487, 2652, 2764, 2959])

# Get a sub graph only containing the stocks from cluster_centers_indices
G = G.subgraph(np.array(G.nodes)[cluster_centers_indices])

pos = nx.spring_layout(G)

# Draw edges
nx.draw_networkx_edges(G, pos=pos, alpha=.1, ax=ax[0])
# Draw labels
nx.draw_networkx_labels(G, pos=pos, font_size=7, ax=ax[0])


percentage_change_range = 11
red2gray = list(Color('red').range_to(Color('gray'), percentage_change_range))
gray2green = list(Color('gray').range_to(Color('green'), percentage_change_range-5))

def animate(year):
    # Set title at top of plot to week number
    # fig.suptitle(f'Yearly change)')
    fig.suptitle(f'Year {dataframes_combined_yearly.index[year]}')

    first_stockprices = dataframes_[year].iloc[0]
    last_stockprices = dataframes_[year].iloc[-1]
    
    # Calculate percentage change
    change = ((last_stockprices - first_stockprices) / first_stockprices) * 100
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
    A = corr_[year]
    # G = nx.read_gexf(f'./data/stockcorr_t{threshold}.gexf')
    #standerd scaling
    new_G = nx.from_numpy_matrix(A, create_using=nx.Graph)
    # Get only first x nodes
    # new_G = new_G.subgraph(stocks_to_include)
    new_G = new_G.subgraph(np.array(new_G.nodes)[cluster_centers_indices])
    # Copy new_G to prevent frozen graph
    new_G = new_G.copy()
    new_G = sc.relabel_graph(new_G, dataframes_[year].columns) # relabel nodes  

    # Clear ax[0] (top subplot)
    ax[0].clear()

    # Draw the edges of the new graph
    nx.draw_networkx_edges(new_G, pos=pos, alpha=.1, ax=ax[0])
    
    # Draw labels
    nx.draw_networkx_labels(new_G, pos=pos, font_size=7, ax=ax[0])
         
    # Draw nodes
    nx.draw_networkx_nodes(new_G, pos=pos, node_color=colors, node_size=20, ax=ax[0])

    # Plot current data on ax[1]
    ax[1].lines[0].set_data(dataframes_combined_yearly['Close'][:year+1].index, dataframes_combined_yearly['Close'][:year+1])
    # Plot dot on every year
    ax[1].scatter(dataframes_combined_yearly['Close'][:year+1].index, dataframes_combined_yearly['Close'][:year+1], s=1, color='darkblue')
    # Plot dot on future weeks
    ax[1].scatter(dataframes_combined_yearly['Close'][year+1:].index, dataframes_combined_yearly['Close'][year+1:], s=1, color='lightgray')

    # Set y axis label
    ax[1].set_ylabel('NASDAQ (USD)')
    # Tilt ax[1] x-labels
    plt.setp(ax[1].get_xticklabels(), rotation=90,)

    return fig


ani = animation.FuncAnimation(fig, animate, frames=years, interval=1000, repeat=True)
# plt.show()

# Save as gif
ani.save('yearly_change.gif', writer='imagemagick', fps=1)