import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from utils.visualize import viz

RANDOM_STATE = 42


def random_forest(df, features=False):
    """

    :param features:
    :param df:
    :return:
    """

    X, y = df.drop('price', axis='columns'), df['price']
    X = pd.get_dummies(X, drop_first=True)

    # scale y
    y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=RANDOM_STATE)
    rf = hyper_params(X_train, y_train)

    sort = None

    if features:
        sort = choose_features(rf, X_train, y_train)

    return rf, sort


def return_tree():
    pass


def hyper_params(x, y, name='rf'):
    """

    :param x:
    :param y:
    :param name:
    :return:
    """

    if name == 'rf':
        # define a model
        rf = RandomForestRegressor()

        # define parameters to run
        params_rf = {
            'n_estimators': [300, 400, 500],
            'max_depth': [80, 90, 100, 110],
            'min_samples_leaf': [3, 4, 5],
            'max_features': ['log2', 'sqrt'],
            'bootstrap': [True]
        }

        # find best parameters
        grid = GridSearchCV(estimator=rf, param_grid=params_rf, cv=3,
                            scoring='neg_mean_squared_error',
                            verbose=1, n_jobs=-1, refit=True).fit(x, y)
        params = grid.best_params_
        print('RANDOM FOREST best params:', params)

        return RandomForestRegressor(params['n_estimators'],
                                     max_depth=params['max_depth'],
                                     min_samples_leaf=params['min_samples_leaf'],
                                     max_features=params['max_features'],
                                     bootstrap=True,
                                     random_state=RANDOM_STATE)

    if name == 'xgboost':
        param_test2 = {
            'max_depth': [2, 5, 10],
            'min_child_weight': [4, 5, 6],
            'n_estimators': [100, 200, 300]
        }

        grid = GridSearchCV(estimator=xgb.XGBRegressor(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                       min_child_weight=2, gamma=0, subsample=0.8,
                                                       colsample_bytree=0.8, objective='reg:squarederror',
                                                       nthread=4, scale_pos_weight=1, seed=RANDOM_STATE),
                            param_grid=param_test2, scoring='roc_auc', n_jobs=4, cv=5).fit(x, y)
        params = grid.best_params_
        print('XGBOOST best params:', params)
        return xgb.XGBRegressor(learning_rate=0.1,
                                n_estimators=params['n_estimators'],
                                max_depth=params['max_depth'],
                                min_child_weight=params['min_child_weight'],
                                gamma=0, subsample=0.8,
                                colsample_bytree=0.8, objective='reg:squarederror',
                                nthread=4, scale_pos_weight=1, seed=RANDOM_STATE),

    if name == 'adaboost':
        dt = DecisionTreeRegressor(max_depth=1, random_state=RANDOM_STATE)
        hyperparameter_space = {'n_estimators': list(range(2, 102, 2)),
                                'learning_rate': np.arange(0.1, 1, 0.1)}

        grid = GridSearchCV(AdaBoostRegressor(base_estimator=dt,
                                              random_state=RANDOM_STATE),
                            param_grid=hyperparameter_space,
                            scoring='neg_mean_squared_error', n_jobs=-1, cv=5).fit(x, y)
        params = grid.best_params_
        print('ADABOOST best params:', params)
        return AdaBoostRegressor(dt, n_estimators=params['n_estimators'],
                                 learning_rate=params['learning_rate'],
                                 random_state=RANDOM_STATE)


def choose_features(model, x, y):
    model.fit(x, y)

    # get and sort feature importances
    importance = pd.DataFrame(model.feature_importances_, index=x.columns).rename(columns={0: 'importance'})
    sort = importance.sort_values('importance', ascending=False).reset_index().head(15)
    viz.show_importance(sort)

    return sort
