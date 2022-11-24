def hist_volatility(single_stock):
    r = np.diff(np.log(single_stock.dropna()))
    mean = np.mean(r)
    residuals_sq = [(r[i] - mean)**2 for i in range(len(r))]
    std_ = np.sqrt(np.sum(residuals_sq)/(len(r)-1))
    vola = std_*np.sqrt(len(single_stock.dropna()))
    return vola    

def get_all_vol(stockdf):
    all_vola = pd.DataFrame(columns=stockdf.columns)
    temp_list = []
    for col in stockdf.columns:
        temp_list.append(hist_volatility(stockdf[col]))
    all_vola.loc[0] = temp_list
    print(f'first 10 from all_vola:{all_vola.iloc[0:10]}')
    return all_vola

all_vola = get_all_vol(stockdf)

## plot stockdf network with color based on volatility with higher volatility being in the center
pos = []*len(all_vola.columns)
for i in range(len(all_vola.columns[0])):
    pos[i] =  (all_vola.values[i], all_vola.values[i])
nx.draw(G, node_color=np.sqrt(1-all_vola.values), node_size=10, with_labels=False)

## get the standard deviation of all the stocks in stockdf
for i in stockdf.columns:
    std_ = np.std(stockdf[i])
    if np.std(stockdf[i]) == 0:
        print(f'{i} has a standard deviation of {np.std(stockdf[i])}')


nx.draw(G)
 
thresh = 0