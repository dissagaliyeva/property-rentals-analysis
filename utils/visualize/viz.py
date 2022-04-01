import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def corr_matrix(df, col=None):
    if col is None:
        col = 'price'

    assert col in df.columns, 'Invalid column'

    if len(df.columns) < 10:
        corr = df.corr()
        mask = np.array(corr)
        mask[np.tril_indices_from(mask)] = False

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(f'{col.title()} column correlation', fontsize=20)
        sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True, fmt='.1g', ax=ax[0])
        single_corr(df, col, ax=ax[1])
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        single_corr(df, col, ax=ax)
        plt.show()


def single_corr(df, col, ax):
    # sorted column representation
    return sns.heatmap(df.corr()[[col]].sort_values(by=col, ascending=False)[1:],
                       annot=True, vmin=-1, vmax=1, ax=ax)


def visualize(df, col, title, min_=None, max_=None, sort=True):
    temp = get_df(df, col, min_, max_)

    if sort:
        temp = temp.groupby(col).size().sort_values(ascending=False)
    else:
        temp = temp[col].value_counts().reset_index().sort_values(by='index')

    temp.plot(x='index', y=col, kind='bar', rot=60).set_title(title)
    plt.show()


def get_df(df, col, min_, max_):
    if min_ and max_:
        if min_ < 0 or max_ < min_ or max_ < 0: return df
        return df[(df[col] >= min_) & (df[col] <= max_)]
    if min_:
        if min_ < 0: return df
        return df[df[col] >= min_]
    if max_:
        if max_ < 0: return df
        return df[df[col] <= max_]
    return df


def plot_feature_importance(feature_importance, title, feature_names):
    # Normalize the importance values
    feature_importance = 100.0 * (feature_importance / max(feature_importance))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importance))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure(figsize=(15, 5))
    plt.bar(pos, feature_importance[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


def property_room(df, col='bedrooms', val=0):
    if type(val) == tuple:
        temp = df[(df[col] > val[0]) | (df[col] <= 0)]
    else:
        temp = df[df[col] == val]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{col.title()}={val} distribution', fontsize=20)

    sns.countplot(y=temp['property_type'], ax=ax[0],
                  order=temp['property_type'].value_counts().index)

    sns.countplot(y=temp['room_type'], ax=ax[1],
                  order=temp['room_type'].value_counts().index)
    plt.tight_layout()
    plt.show()


def hist(df, col, bins=30):
    title = col.title() + ' distribution'

    plt.figure(figsize=(7, 5))
    df[col].plot(kind='hist', bins=bins).set_title(title)
    plt.xlabel(col)
    plt.show()


def get_property(df, n, head=True):
    temp = df['property_type'].value_counts()

    if head:
        temp = temp.head(n).reset_index()
    else:
        temp = temp.tail(n).reset_index()

    if temp['property_type'].mean(axis=0) == 1:
        print('Cannot plot a KDE plot, all values are 1\n\n')
        print(temp)
    else:
        temp = temp['index'].tolist()
        temp = df.drop(index=df[~df['property_type'].isin(temp)].index, axis='columns')
        kdeplot(temp)
        return temp


def kdeplot(df):
    plt.figure(figsize=(15, 10))
    sns.kdeplot(data=df, x='price', hue='property_type')
    plt.legend(df['property_type'].unique().tolist())
    plt.show()


def show_insights(df, col, prop):
    temp = df[df[col] == prop]
    temp[['price']].boxplot().set_title(f'{prop.title()} price distribution')
    plt.show()

    print(temp.price.describe())
    mean, iqr = np.mean(temp['price']), stats.iqr(temp['price'])
    print(f'\nOutlier (mean +- 1.5*IQR)= [{mean - 1.5 * iqr}, {mean + 1.5 * iqr}]')