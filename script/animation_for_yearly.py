import networkx as nx
import numpy as np
import pandas as pd
from colour import Color
from matplotlib import animation
from matplotlib import pyplot as plt

from stockcorr import get_corr_matrix, relabel_graph
import collective_weight_functions as cwf

# Animate stocks so they go green if their price increases and red if it decreases
years = 18
# Load stock data csv
# Format is: stocks as columns and dates as rows
stockdf = pd.read_csv('../data/stock_market_data/stockdf.csv')

corr_, total_sum, dataframes_ = cwf.split_into_years(stockdf)


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
A = corr_[0]
# G = nx.read_gexf(f'./data/stockcorr_t{threshold}.gexf')
G = nx.from_numpy_matrix(A, create_using=nx.Graph)
G = relabel_graph(G, dataframes_[0].columns) # relabel nodes

# Get only first x nodes
G = G.subgraph(list(G.nodes)[:100])

pos = nx.spring_layout(G)

G.nodes

# Draw edges
# TODO: (Cumulative) Redraw edges every frame depending on correlation for that week?
nx.draw_networkx_edges(G, pos=pos, alpha=.1, ax=ax[0])
# Draw labels
nx.draw_networkx_labels(G, pos=pos, font_size=7, ax=ax[0])


percentage_change_range = 11
red2gray = list(Color('red').range_to(Color('gray'), percentage_change_range))
gray2green = list(Color('gray').range_to(Color('green'), percentage_change_range-5))

def animate(year):
    # Set title at top of plot to week number
    fig.suptitle(f'Yearly change)')
    # Get stock prices for this week
    stockprices = dataframes_[year]

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
        print(pc)     
        # Get color
        if pc > 0:
            color = gray2green[min(percentage_change_range-1-5, int(pc))]
            colors.append(color.hex)
        elif pc < 0:
            color = red2gray[min(percentage_change_range-1, int(abs(pc)))]
            colors.append(color.hex)
        else:
            colors.append('#808080')

            
    # # Calculate node correlation from first week up until current week 
    # # Step 1: Create df with data from first week up until current week
    # intermediate_df = stockdf.iloc[:week+1]
    # # Step 2: Calculate correlation
    # numpy_correlations = intermediate_df.corr().to_numpy()
    # # Step 3: Cutoff correlation values with threshold
    # if threshold != 0:
    #     # Where the absolute value of the correlation is smaller than the threshold, set it to 0
    #     numpy_correlations = np.where(abs(numpy_correlations) > threshold, numpy_correlations, 0)
    # # Where the correlation is 1, set it to 0
    # numpy_correlations = np.where(numpy_correlations == 1, 0, numpy_correlations)
    # # TODO: Standard scale?
    # # mask = numpy_correlations != 0
    # # numpy_correlations[mask] = (numpy_correlations[mask] - numpy_correlations[mask].mean()) / numpy_correlations[mask].std()

    # # Step 4: Create graph from correlation matrix
    # new_G = nx.from_numpy_matrix(numpy_correlations, create_using=nx.Graph)
    # # Get only first x nodes
    # new_G = new_G.subgraph(list(new_G.nodes)[:100])
    # # new_G = relabel_graph(new_G, stockdf.columns) # relabel nodes
            
    # Get correlation matrix
    A = get_corr_matrix(stockdf.iloc[:week+1], threshold=threshold)
    # G = nx.read_gexf(f'./data/stockcorr_t{threshold}.gexf')
    #standerd scaling
    mask = A != 0
    A[mask] = (A[mask] - A[mask].mean()) / A[mask].std()
    new_G = nx.from_numpy_matrix(A, create_using=nx.Graph)
    # Get only first x nodes
    new_G = new_G.subgraph(list(new_G.nodes)[:100])
    # Copy new_G to prevent frozen graph
    new_G = new_G.copy()
    new_G = relabel_graph(new_G, stockdf.columns) # relabel nodes  

    # Clear ax[0] (top subplot)
    ax[0].clear()

    # Draw the edges of the new graph
    nx.draw_networkx_edges(new_G, pos=pos, alpha=.1, ax=ax[0])
    
    # Draw labels
    nx.draw_networkx_labels(new_G, pos=pos, font_size=7, ax=ax[0])


    # nx.draw_networkx_edges(new_G, pos=pos, alpha=.1, ax=ax[0])

    
            
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


ani = animation.FuncAnimation(fig, animate, frames=years, interval=500, repeat=True)
plt.show()