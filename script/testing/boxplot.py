import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib as mpl
import numpy as np 
import sys
sys.path.append('./script/data_clean/')
import newClean as nc

def boxplot_plotter(clusters, corr_df, year):
    plt.close('all')
    
    fig = plt.figure(figsize =(30, 30))# create figure & 1 axis 
    ax = fig.add_subplot(111)

    all_data = []
    n_labels = clusters.labels_.max()
    for y in range(n_labels+1):
        data = []
        for i in np.argwhere(clusters.labels_ == y):
            data.append(corr_df[i, y][0])
        all_data.append((y,data))
        
    ## sort thedata by median and return the original index
    all_data = sorted(all_data, key=lambda x: np.median(x[1]))
    original_index = [x[0] for x in all_data]
    all_data = [x[1] for x in all_data]

    print("max index", max(original_index))

    print("amount of clusters", len(clusters.cluster_centers_indices_))

    bp = plt.boxplot(all_data, patch_artist=True)

    print("average of all clusters", np.average([len(i) for i in all_data]))
    for box in bp["boxes"]:
        box.set( facecolor='gray')
    for whisker in bp["whiskers"]:
        whisker.set(color='orange', linewidth=2)
    for cap in bp["caps"]:
        cap.set(color='black', linewidth=2)
    for median in bp["medians"]:
        median.set(color='white', linewidth=2)
    for flier in bp["fliers"]:
        flier.set(marker='o', color='red')
    
    ax.set_title(f'Boxplot of the Affinity Propagation Clusters of {year}', fontsize=30)
    ax.set_xlabel('Clusters', fontsize=20)
    ax.set_ylabel('Correlation', fontsize=20)
    ax.set_ylim(-0.5,0.9)
    ## remove all x_ticks and labels
    ax.xaxis.set_ticklabels([])
    plt.show()


def main():
    df_years, min_year, max_year = nc.get_data()
    clusters = nc.read_affinity_propagation_from_year(2020, debug=True)
    corr_df = nc.get_corr_from_year(2020, df_years,min_year)
    boxplot_plotter(clusters, corr_df, 2020)

if __name__ == "__main__":
    main()