import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from kmeans import kmeans

DATA_DIR = Path('CSC2042S-Assignment1-Data')
WDI_PATH = DATA_DIR / 'WDICSV.csv'


def load_wdi_dataset(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code']
    year_cols = [c for c in raw.columns if c.isdigit()]
    tidy = raw.melt(id_vars=id_vars, value_vars=year_cols,
                    var_name='Year', value_name='Value').dropna(subset=['Value'])
    tidy['Year'] = tidy['Year'].astype(int)
    pivot = tidy.pivot_table(index=['Country Name','Country Code','Year'],
                             columns='Indicator Code', values='Value').reset_index()
    return pivot


def preprocess(df: pd.DataFrame, feature_thresh: float=0.3, sample_thresh: float=0.7) -> pd.DataFrame:
    feature_coverage = df.notna().mean()
    df = df.loc[:, feature_coverage >= feature_thresh]
    sample_coverage = df.notna().mean(axis=1)
    df = df.loc[sample_coverage >= sample_thresh]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    numeric = df.select_dtypes(include=[np.number])
    scaled = scaler.fit_transform(numeric)
    df[numeric.columns] = scaled
    return df


def run_experiment(X: np.ndarray, k: int, init: str, runs: int = 5):
    results = []
    for seed in range(runs):
        centroids, labels, inertia, n_iter, history = kmeans(X, k, init=init, random_state=seed)
        results.append((inertia, n_iter, history))
    return results


def summarize(results):
    inertias = [r[0] for r in results]
    iters = [r[1] for r in results]
    return np.mean(inertias), np.std(inertias), np.mean(iters)


def main():
    df = load_wdi_dataset(WDI_PATH)
    df = preprocess(df)
    df = normalize(df)
    features = df.drop(columns=['Country Name','Country Code','Year'])
    X = features.values
    k = 5
    for init in ['random','kmeans++']:
        results = run_experiment(X, k, init, runs=5)
        mean_inertia, std_inertia, mean_iter = summarize(results)
        print(f"Initialization: {init}")
        print(f"  Mean inertia: {mean_inertia:.4f} ± {std_inertia:.4f}")
        print(f"  Mean iterations: {mean_iter:.2f}")

        # Plot convergence of first run
        history = results[0][2]
        plt.plot(history)
        plt.title(f"Convergence ({init})")
        plt.xlabel('Iteration')
        plt.ylabel('Inertia')
        plt.savefig(f'convergence_{init}.png')
        plt.clf()


if __name__ == '__main__':
    main()
