import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from utils.visualize import viz

RANDOM_STATE = 42


def get_features(df):
    """
        Create a tuned RandomForestRegressor to select most important
        features. This function takes care of X and y, as well as
        train/test division.

        Parameters
        ----------

        df  :   Pandas DataFrame object
                Pandas DataFrame object containing all 10 columns,
                including the target, price, column.

        Returns
        -------
        Returns a sorted DataFrame object containing column names
        and their feature importance scores ranging from 0 to 1.
        Their cumulative sum must be equal to 1.
    """

    # create x and y divisions
    X, y = df.drop('price', axis='columns'), df['price']

    # encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # create train/test values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=RANDOM_STATE)

    # fine-tune the model
    rf = hyper_params(X_train, y_train)

    # get sorted feature importances
    sort = choose_features(rf, X_train, y_train)

    return sort


def choose_features(model, x, y):
    """
        Create a DataFrame object from a specified model that stores
        column names and their respective feature importance scores.
        It truncates to only 15 most important values (descending order).

        Returns a sorted DataFrame object storing features and scores.

        Parameters
        ----------

        model   :   sklearn algorithm (default: RandomForestRegressor)
        x       :   Features of a dataset (without target column)
        y       :   Target of a dataset (without feature columns)


        Returns
        -------
        A sorted DataFrame containing columns and respective feature
        importance scores.
    """

    # fit the parameters
    model.fit(x, y)

    # get and sort feature importances
    importance = pd.DataFrame(model.feature_importances_, index=x.columns).rename(columns={0: 'importance'})

    # sort values
    sort = importance.sort_values('importance', ascending=False).reset_index().head(15)

    # visualize
    viz.show_importance(sort)

    return sort


def hyper_params(x, y, name='rf'):
    """
        Tune one of the three models using the specified
        dictionary of parameters. Use GridSearchCV to test
        the model and select the best parameters.

        Returns a fitted, fine-tuned model.

        Parameters
        ----------
            x    :      pandas DataFrame
                        Features of a dataset (without target column)

            y    :      pandas Series
                        Target of a dataset (without feature columns)

            name :      string (default: 'rf')
                        Name of the model. It could be one of three:
                        - "rf" for RandomForestRegressor
                        - "adaboost" for AdaboostRegressor
                        - "xgboost" for XGBoostRegressor

        Returns
        -------
        A fitted, fine-tuned model.
    """

    # verify model's correctness
    assert name in ['rf', 'adaboost', 'xgboost'], 'Invalid model'

    # create a RandomForestRegressor model
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

        # select best parameters
        params = grid.best_params_
        print('RANDOM FOREST best params:', params)

        # refit the model
        return RandomForestRegressor(params['n_estimators'],
                                     max_depth=params['max_depth'],
                                     min_samples_leaf=params['min_samples_leaf'],
                                     max_features=params['max_features'],
                                     bootstrap=True,
                                     random_state=RANDOM_STATE)

    # create a XGBRegressor model
    if name == 'xgboost':

        # define parameters to run
        param_test2 = {
            'max_depth': [2, 5, 10],
            'min_child_weight': [4, 5, 6],
            'n_estimators': [100, 200, 300]
        }

        # find best parameters
        grid = GridSearchCV(estimator=xgb.XGBRegressor(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                       min_child_weight=2, gamma=0, subsample=0.8,
                                                       colsample_bytree=0.8, objective='reg:squarederror',
                                                       nthread=4, scale_pos_weight=1, seed=RANDOM_STATE),
                            param_grid=param_test2, scoring='roc_auc', n_jobs=4, cv=5).fit(x, y)

        # select best parameters
        params = grid.best_params_
        print('XGBOOST best params:', params)

        # refit the model
        return xgb.XGBRegressor(learning_rate=0.1,
                                n_estimators=params['n_estimators'],
                                max_depth=params['max_depth'],
                                min_child_weight=params['min_child_weight'],
                                gamma=0, subsample=0.8,
                                colsample_bytree=0.8, objective='reg:squarederror',
                                nthread=4, scale_pos_weight=1, seed=RANDOM_STATE),

    # create a Adaboost model
    if name == 'adaboost':

        # define parameters to run
        dt = DecisionTreeRegressor(max_depth=1, random_state=RANDOM_STATE)
        hyperparameter_space = {'n_estimators': list(range(2, 102, 2)),
                                'learning_rate': np.arange(0.1, 1, 0.1)}

        # find best parameters
        grid = GridSearchCV(AdaBoostRegressor(base_estimator=dt,
                                              random_state=RANDOM_STATE),
                            param_grid=hyperparameter_space,
                            scoring='neg_mean_squared_error', n_jobs=-1, cv=5).fit(x, y)

        # select best parameters
        params = grid.best_params_
        print('ADABOOST best params:', params)

        # refit the model
        return AdaBoostRegressor(dt, n_estimators=params['n_estimators'],
                                 learning_rate=params['learning_rate'],
                                 random_state=RANDOM_STATE)


