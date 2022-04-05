import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def build_ols(df, scale='standard', verbose=True):
    X, y = get_xy(df)

    scaler = StandardScaler() if scale == 'standard' else MinMaxScaler()
    X = scaler.fit_transform(X)

    Xc = sm.add_constant(X)
    linreg = sm.OLS(y, Xc).fit()
    if verbose:
        print(linreg.summary())
    else:
        print('R2:', round(linreg.rsquared, 2))


def build_linreg(df, norm=True, scale='standard', show_results=True):
    X, y = get_xy(df)

    if norm:
        scaler = StandardScaler() if scale == 'standard' else MinMaxScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    linreg = LinearRegression(normalize=False).fit(X_train, y_train)

    if show_results:
        print_results(linreg, X_train, X_test, y_train, y_test)
        
    return linreg


def print_results(model, X_train, X_test, y_train, y_test):
    output = model.predict(X_test)
    print(f'MSE (train): {mean_squared_error(y_train, model.predict(X_train))}')
    print(f'MSE (test): {mean_squared_error(y_test, output)}')
    print('=====')
    print(f'MAE (train): {mean_absolute_error(y_train, model.predict(X_train))}')
    print(f'MAE (test): {mean_absolute_error(y_test, output)}')
    print('=====')
    print(f'Score (train): {model.score(X_train, y_train)}')
    print(f'Score (test): {model.score(X_test, y_test)}')


def get_xy(df):
    X, y = pd.get_dummies(df.drop('price', axis='columns'), drop_first=True), df['price'].values
    y = np.log1p(y)
    return X, y
