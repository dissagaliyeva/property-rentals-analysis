import numpy as np
import seaborn as sns
import ppscore as pps
import matplotlib.pyplot as plt


def property_room(df, col='bedrooms', val=None):
    """
        Visualize pandas DataFrame's column distribution.
        It's possible to select DataFrame's subset at a
        specific value. See Examples for more details.

        Parameters
        ----------

        df  :   Pandas DataFrame object

        col :   string
                Column from a DataFrame object to subset

        val :   int or tuple
                Number or tuple of numbers to subset a
                DataFrame. See Examples for more details

        Examples
        --------
        >>> property_room(df, 'bedrooms', (3, 9))
        This line will show all entries where bedrooms
        bigger than 3 and less than 9 (including).

        >>> property_room(df, 'bathrooms', 1)
        This line will show all entries where bathrooms
        are equal to 1.
    """

    # select necessary entries
    if type(val) == tuple:
        temp = df[(df[col] > val[0]) | (df[col] <= val[1])]
    else:
        temp = df[df[col] == val]

    # create subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # set title
    fig.suptitle(f'{col.title()}={val} distribution', fontsize=20)

    # show a count plot for property type
    sns.countplot(y=temp['property_type'], ax=ax[0],
                  order=temp['property_type'].value_counts().index)

    # show a count plot for room type
    sns.countplot(y=temp['room_type'], ax=ax[1],
                  order=temp['room_type'].value_counts().index)
    plt.tight_layout()
    plt.show()


def show_importance(sort):
    """
        Visualize the resulting RandomForestRegressor features.
        It should contain two columns only: "importance" and "index".

            - "index" column should contain all 34 columns without "price".
            - "importance" column should contain values within 0-1, the
              cumulative sum should be equal to 1.

        Parameters
        ----------

        sort :  Pandas DataFrame object
                Pandas DataFrame object containing two columns: "importance"
                and "index". It is the result of running RandomForestRegressor
                to get most important features.

    """
    # verify the columns are present
    length = len(set(sort.columns.tolist()).difference({'importance', 'index'}))
    assert length == 0, "DataFrame does not have either 'importance' or 'index' columns."

    # visualize importance
    sns.barplot(np.arange(0, len(sort)), sort['importance'])

    # set title
    plt.title('Feature importances', fontsize=17)

    # set ticks and labels
    plt.xticks(np.arange(0, len(sort)), sort['index'], rotation=90)
    plt.yticks(np.arange(0, 0.5, 0.1))
    plt.xlabel('Column', fontsize=14)
    plt.ylabel('Importance', fontsize=14)

    # fix representation and show
    plt.tight_layout()
    plt.show()


def pps_matrix(df):
    """
        Calculate and visualize Predictive Power Score for a
        pandas DataFrame with both numeric and categorical columns.

        Parameters
        ----------

        df :    Pandas DataFrame object
    """

    # calculate pps score
    matrix = pps.matrix(df)

    # Prepare data to pivot table
    pps_pivot = matrix.pivot('x', 'y', 'ppscore')
    pps_pivot.index.name, pps_pivot.columns.name = None, None

    # Plot
    plt.figure(figsize=(10, 4))
    sns.heatmap(pps_pivot, annot=True, cmap='YlGn')

    # set title
    plt.title('Predictive Power Score Matrix', fontsize=17)
    plt.show()
