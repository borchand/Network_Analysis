import json 
import os
import glob
from tqdm import tqdm

def get_col_to_drop():
    datasets = [
        "nyse",
        "nasdaq",
        "sp500",
        "forbes2000"
    ]
    x = []
    for dataset in datasets:
        print(dataset)
        all_files = glob.glob(f"../data/stock_market_data/{dataset}/json/*.json")

        for file in tqdm(all_files):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    temp = data['chart']['result'][0]['meta']['instrumentType']
                    date = data['chart']['result'][0]['meta']['firstTradeDate']
                if temp in ['ETF', 'MUTUALFUND']:
                    x.append(data['chart']['result'][0]['meta']['symbol'])
                elif date == None or date > 1420072583:
                    x.append(data['chart']['result'][0]['meta']['symbol'])
            except Exception as e:
                print(e)

    return x


def main():
    get_col_to_drop()

if __name__ == "__main__":
    main()
