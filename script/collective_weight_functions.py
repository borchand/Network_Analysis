import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import stockcorr as sc
from networkx.algorithms import community


def split_into_years(threshold = 0.9, one_hot_where=False):
    ## IMPORTANT: Only works when index is not the date column
    stock_df = pd.read_csv('../data/stock_market_data/stockdf.csv', index_col=0)
    stock_df['Date'] = pd.to_datetime(stock_df.index)
    first_date = stock_df['Date'].dt.year.drop_duplicates()

    ## split the stockdf into dataframes, one for each year

    dataframes_ = [stock_df[stock_df['Date'].dt.year == year] for year in first_date]
    dataframes_ = [df.set_index('Date') for df in dataframes_]
    corr_list = []
    for i, dataframe in enumerate(dataframes_):
        print(f"getting corr matrix {i} out of {len(dataframes_)}", end = "\r")
        curr_ = sc.get_corr_matrix(df=dataframe, threshold=threshold, verbose=False)
        if one_hot_where:
            # make every value to 0 if below 0
            curr_ = np.where(curr_ > 0, curr_ , 0)
        else:
            ## make every value to 1 if above 0        
            curr_ = np.where(curr_ > 0, 1 , 0)
            
        corr_list.append(curr_)

    ## sum all correlation matrices in corr_list



    total_sum = np.sum(corr_list, axis=0)
    return corr_list, total_sum, dataframes_


def soting_indexes(list_of_lists:list[list]):
    intersects = []

    for i in range(len(list_of_lists)):
        for j in range(i+1,len(list_of_lists)):
            set_i = set(list_of_lists[i])
            set_j = set(list_of_lists[j])
            intersect = set_i.intersection(set_j)
            intersects.append([list(intersect),i,j])

    idx_in_list = list(range(len(list_of_lists)))
    # print(idx_in_list)
    
    # sort intersect by length of the first element in the list
    intersects = sorted(intersects, key=lambda x: len(x[0]), reverse=True)
    # print(intersects)


    final_list = []
    stored_idx = None
    while idx_in_list != []:
        if stored_idx == None:
            final_list.append(list_of_lists[intersects[0][1]])
            final_list.append(list_of_lists[intersects[0][2]])
            if intersects[0][1] in idx_in_list:
                idx_in_list.remove(intersects[0][1])
            if intersects[0][2] in idx_in_list:
                idx_in_list.remove(intersects[0][2])
            stored_idx = intersects[0][2]
            intersects.pop(0)

        else:
            found = False
            for i in range(len(intersects)):
                if stored_idx == intersects[i][1]:
                    final_list.append(list_of_lists[intersects[i][2]])
                    if intersects[i][2] in idx_in_list:
                        idx_in_list.remove(intersects[i][2])
                    found = True
                    break
                
                elif stored_idx == intersects[i][2]:
                    final_list.append(list_of_lists[intersects[i][1]])
                    if intersects[i][1] in idx_in_list:
                        idx_in_list.remove(intersects[i][1])
                    found = True

                    intersects.pop(i)
                    break
            if not found:
                stored_idx = None
                    
    # print(final_list)
    ## flatten final_list
    final_list = [item for sublist in final_list for item in sublist]

    final_list = list(dict.fromkeys(final_list))

    return final_list



def main():
    list_ = [[5,6,7,8,9,4,3], [1,2,3,4,9], [1,2,3,88], [11,12,13,14,10,9,8,7,1,2,3]]

    soting_indexes(list_)

if __name__ == '__main__':
    main()


        
