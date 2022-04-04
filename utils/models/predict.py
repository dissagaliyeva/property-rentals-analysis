from utils.models import quick_test
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def sample_predict(df, n=1):
    df_sample = df.sample(n=n)
    temp = df_sample.copy()

    X, y = prepare_samples(df_sample)

    # current linear regression
    lin_reg = quick_test.build_linreg(df, norm=True, show_results=False)
    yhat = lin_reg.predict(X)

    temp['predicted'] = np.expm1(yhat)
    temp['difference (actual - predicted)'] = temp['price'] - temp['predicted']
    return temp


def prepare_samples(df_sample):
    # prepare the sample
    sample_X, sample_y = df_sample.drop('price', axis='columns'), df_sample['price']
    sample_X = pd.get_dummies(sample_X, drop_first=True)

    scaler = StandardScaler()
    sample_Xst = scaler.fit_transform(sample_X)
    return sample_Xst, sample_y



