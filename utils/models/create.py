import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from utils.visualize import viz


def random_forest(df, features=False):
    """

    :param features:
    :param df:
    :return:
    """

    rf = RandomForestRegressor()

    X, y = df.drop('price', axis='columns'), df['price']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=101)
    best = hyper_params(rf, X_train, y_train).best_params_
    print('Best params:', best)

    rf = RandomForestRegressor(best['n_estimators'],
                                 max_depth=best['max_depth'],
                                 min_samples_leaf=best['min_samples_leaf'],
                                 max_features=best['max_features'],
                                 bootstrap=True)

    if features:
        sort = choose_features(rf, X_train, y_train)

    return rf, sort


def hyper_params(model, x, y, name='rf'):
    """

    :param model:
    :param x:
    :param y:
    :param name:
    :return:
    """

    if name == 'rf':
        # define parameters to run
        params_rf = {
            'n_estimators': [300, 400, 500],
            'max_depth': [80, 90, 100, 110],
            'min_samples_leaf': [3, 4, 5],
            'max_features': ['log2', 'sqrt'],
            'bootstrap': [True]
        }

        # find best parameters
        grid_rf = GridSearchCV(estimator=model, param_grid=params_rf, cv=3,
                               scoring='neg_mean_squared_error',
                               verbose=1, n_jobs=-1, refit=True).fit(x, y)
        return grid_rf

    if name == 'xgb':
        pass


def choose_features(model, x, y):
    model.fit(x, y)

    # get and sort feature importances
    importance = pd.DataFrame(model.feature_importances_, index=x.columns).rename(columns={0: 'importance'})
    sort = importance.sort_values('importance', ascending=False).reset_index().head(15)
    viz.show_importance(sort)

    return sort




