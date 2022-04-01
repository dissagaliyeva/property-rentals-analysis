import numpy as np
import pandas as pd


def describe_outliers(df, col):
    assert col in df.columns, 'Invalid column name'

    q1, mean, q3 = np.quantile(df[col], [.25, .5, .75])
    iqr = q3 - q1

    suspects_range = [mean - 1.5 * iqr, mean + 1.5 * iqr]
    suspects = df[df[col] > suspects_range[1]]

    outliers_range = [mean - 3.0 * iqr, mean + 3.0 * iqr]
    outliers = df[df[col] > outliers_range[1]]

    print(f'{col.title()} suspects (mean +- 1.5 * IQR) at:', suspects_range)
    print('Number of suspects:', len(suspects))

    print('\n====\n')

    print(f'{col.title()} definite (mean +- 3.0 * IQR) outliers at:', outliers_range)
    print('Number of outliers:', len(outliers))

    return suspects, outliers


def calc_mad(x, threshold=3.5):
    x = x[:, None]

    median = np.median(x, axis=0)
    diff = np.sum((x - median)**2, axis=-1)
    diff = np.sqrt(diff)
    mad = np.median(diff)
    z_score = 0.6745 * diff / mad

    return z_score > threshold


def eigenvalues(df):
    features, target = df.drop('price', axis='columns'), df['price']
    features = pd.get_dummies(features, drop_first=True)

    corr = np.corrcoef(features, rowvar=0)
    eigenvalues, eigenvectors = np.linalg.eig(corr)
    print(eigenvalues)

    print(np.where(eigenvalues < 0.01))
    print([round(x, 3) for x in eigenvectors[:, 1]])

    print(return_eigen(eigenvectors, 5), return_col(features, 5))
    print(return_eigen(eigenvectors, 7), return_col(features, 7))
    print(return_eigen(eigenvectors, 12), return_col(features, 12))
    print(return_eigen(eigenvectors, 16), return_col(features, 16))
    print(return_eigen(eigenvectors, 20), return_col(features, 20))


def return_eigen(values, idx):
    return values[:, 1][idx]


def return_col(values, idx):
    return values.columns.tolist()[idx]

# eigenvalues(rentals)