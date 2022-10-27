import pandas as pd

def main():
    df = pd.read_csv('../data/AllfileBig.csv')
    with open('../data/AllfileBig.txt', 'w') as f:
        f.write(df.to_string())

if __name__ == '__main__':
    main()