import datetime
import joblib
import numpy as np
from loguru import logger
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import SGDRegressor, Lasso, GammaRegressor, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, \
    ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from hyperopt import hp, fmin, Trials, tpe, space_eval

from tkinter import messagebox as mb


def params_space(algo_name):
    if algo_name == 'el':
        return {'alpha': hp.uniform('alpha', 0.1, 10), 'l1_ratio': hp.uniform('l1_ratio', 0, 1),
                'max_iter': hp.choice('max_iter', range(100, 1000))}
    elif algo_name == 'sgd':
        return {'loss': hp.choice('loss', ['squared_error', 'huber', 'epsilon_insensitive']),
                'penalty': hp.choice('penalty', ['l1', 'l2'])}
    elif algo_name == 'lasso':
        return {'alpha': hp.uniform('alpha', 0.1, 10),
                'selection': hp.choice('selection', ['cyclic', 'random']),
                'max_iter': hp.choice('max_iter', range(100, 1000))}
    elif algo_name == 'ridge':
        return {'alpha': hp.uniform('alpha', 0.1, 10), 'max_iter': hp.choice('max_iter', range(100, 1000)),
                'solver': hp.choice('solver', ['svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag'])}
    elif algo_name == 'rf':
        return {'max_depth': hp.choice('max_depth', range(2, 31)),
                'min_samples_split': hp.choice('min_samples_split', range(2, 11)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 11)),
                'bootstrap': hp.choice('bootstrap', [True, False]),
                'n_estimators': hp.choice('n_estimators', range(10, 1001))}
    elif algo_name == 'dt':
        return {'splitter': hp.choice('splitter', ['best']), 'max_depth': hp.choice('max_depth', range(1, 31)),
                'min_samples_split': hp.choice('min_samples_split', range(2, 11)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 11)),
                'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2, 101))}
    elif algo_name == 'gb':
        return {'loss': hp.choice('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
                'n_estimators': hp.choice('n_estimators', range(10, 1001)),
                'min_samples_split': hp.choice('min_samples_split', range(2, 11)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 11)),
                'max_depth': hp.choice('max_depth', range(1, 31))}
    elif algo_name == 'bagging':
        return {'n_estimators': hp.choice('n_estimators', range(10, 1001))}
    elif algo_name == 'adaboost':
        return {'n_estimators': hp.choice('n_estimators', range(10, 1001)),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
                'loss': hp.choice('loss', ['linear', 'square', 'exponential'])}
    elif algo_name == 'etr':
        return {'bootstrap': hp.choice('bootstrap', [True, False]),
                'max_depth': hp.choice('max_depth', range(1, 50)),
                'min_samples_split': hp.choice('min_samples_split', range(2, 11)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 11)),
                'n_estimators': hp.choice('n_estimators', range(1, 300))}
    elif algo_name == 'gamma':
        return {'alpha': hp.uniform('alpha', 0.1, 10), 'max_iter': hp.choice('max_iter', range(100, 1000))}
    elif algo_name == 'svr':
        return {'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
                'C': hp.choice('C', range(1, 50)), 'max_iter': hp.choice('max_iter', range(10, 100))}
    elif algo_name == 'lgb':
        return {'reg_alpha': hp.uniform('reg_alpha', 0.05, 5), 'reg_lambda': hp.uniform('reg_lambda', 0.05, 5),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
                'max_depth': hp.choice('max_depth', range(5, 51)),
                'boosting_type': hp.choice('boosting_type', ['gbdt']),
                'min_child_samples': hp.choice('min_child_samples', range(2, 31)),
                'min_child_weight': hp.uniform('min_child_weight', 0, 10)}
    elif algo_name == 'cab':
        return {'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
                'iterations': hp.choice('iterations', [10]),
                'l2_leaf_reg': hp.choice('l2_leaf_reg', range(1, 31)),
                'bagging_temperature': hp.uniform('bagging_temperature', 0.02, 1),
                'boosting_type': hp.choice('boosting_type', ['Ordered', 'Plain']),
                'depth': hp.choice('depth', range(2, 16)),
                'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
                'random_strength': hp.choice('random_strength', range(1, 102)),
                'verbose': hp.choice('verbose', [False])}
    elif algo_name == 'xgb':
        return {'reg_alpha': hp.uniform('reg_alpha', 0, 1), 'reg_lambda': hp.uniform('reg_lambda', 1, 20),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
                'max_depth': hp.choice('max_depth', range(1, 31)),
                'n_estimators': hp.choice('n_estimators', range(1, 301)),
                'min_child_weight': hp.uniform('min_child_weight', 0.1, 10)}


class DATA:
    def __init__(self, data, export_path, bound_path=None):
        self.data = data
        self.minmax = []
        self.export_path = export_path
        self.bound_path = bound_path
        self.algo_dict = {'el': ElasticNet(), 'sgd': SGDRegressor(), 'lasso': Lasso(), 'ridge': Ridge(),
                          'rf': RandomForestRegressor(), 'dt': DecisionTreeRegressor(),
                          'gb': GradientBoostingRegressor(),
                          'bagging': BaggingRegressor(), 'adaboost': AdaBoostRegressor(), 'etr': ExtraTreesRegressor(),
                          'gamma': GammaRegressor(), 'svr': SVR(), 'lgb': LGBMRegressor(), 'cab': CatBoostRegressor(),
                          'xgb': XGBRegressor()}
        self.algo_dict_optimize = {'el': ElasticNet, 'sgd': SGDRegressor, 'lasso': Lasso, 'ridge': Ridge,
                                   'rf': RandomForestRegressor, 'dt': DecisionTreeRegressor,
                                   'gb': GradientBoostingRegressor,
                                   'bagging': BaggingRegressor, 'adaboost': AdaBoostRegressor,
                                   'etr': ExtraTreesRegressor,
                                   'gamma': GammaRegressor, 'svr': SVR, 'lgb': LGBMRegressor,
                                   'cab': CatBoostRegressor,
                                   'xgb': XGBRegressor}

    def preprocess(self, bound_path=None):
        logger.info('Step preprocess is in progress!')
        target = self.data.iloc[:, -1]
        feature = self.data.iloc[:, :-1]
        if not bound_path:
            mm = MinMaxScaler()
            mm.fit_transform(feature)
            minBound, maxBound = mm.data_min_, mm.data_max_
        else:
            bound_df = pd.read_excel(self.bound_path)
            mm = MinMaxScaler()
            mm.fit_transform(bound_df)
            minBound, maxBound = mm.data_min_, mm.data_max_
            mm.transform(feature)
        data = pd.concat([feature, target], axis=1)
        return data, minBound, maxBound

    def split_data(self, feature, target, train_size=0.8):
        logger.info('Step data split is in progress!')
        X, X1, y, y1 = train_test_split(feature, target, train_size=train_size, shuffle=True, random_state=42)
        train_feature = X.values
        train_target = y.values
        valid_feature = X1.values
        valid_target = y1.values
        return train_feature, train_target, valid_feature, valid_target

    def filter_model(self, X_train, X_test, y_train, y_test):
        logger.info('Step filter is in progress!')
        algo_dict = self.algo_dict

        algo = [k for k in algo_dict.keys()]
        model = []
        accuracy_test = []
        accuracy_train = []
        rmse_train = []
        rmse_test = []
        for i in algo:
            logger.info(f'Now algorithm {i} is in filter process')
            algo_dict[i].fit(X_train, y_train.ravel())
            accuracy_train.append(algo_dict[i].score(X_train, y_train.ravel()))
            accuracy_test.append(algo_dict[i].score(X_test, y_test.ravel()))
            train_pred = algo_dict[i].predict(X_train)
            test_pred = algo_dict[i].predict(X_test)
            rmse_train.append(round(mean_squared_error(y_train, train_pred), 1))
            rmse_test.append(round(mean_squared_error(y_test, test_pred), 1))
            model.append(i)

        mod = pd.DataFrame([model, accuracy_train, accuracy_test, rmse_train, rmse_test]).T
        mod.columns = ['model', 'score_train', 'score_test', 'rmse_train', 'rmse_test']
        logger.info(f'filter result:\n{mod}')

        models = mod.sort_values(by='score_test', ascending=False).iloc[0:3, 0].tolist()

        return models

    def hyper_parameter(self, model_list, X, X1, y, y1):
        logger.info('Step hyperparameter optimization is in progress!')
        logger.warning(
            'if catboost is in algorithm list, notice that we set iteration as 10 to speed up there, but in final test progress, it is 100')
        model_after_params_optimize = defaultdict(dict)
        feature, value = np.vstack((X, X1)), np.hstack((y, y1))

        def hyperOn(params):
            all_r2_score = 0
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for train_idx, test_idx in kf.split(feature):
                train_x, test_x = feature[train_idx], feature[test_idx]
                train_y, test_y = value[train_idx], value[test_idx]
                std = model(**params)
                std.fit(train_x, train_y)
                all_r2_score += r2_score(test_y, std.predict(test_x))
            return -all_r2_score

        for mod in model_list:
            logger.info(f'Now algorithm {mod} is in hyperparameter optimization progress')
            model = self.algo_dict_optimize[mod]
            trials = Trials()
            res = fmin(hyperOn, space=params_space(mod), trials=trials, algo=tpe.suggest,
                       max_evals=100)
            res = space_eval(params_space(mod), res)
            logger.info(f'the best params of {mod} is {res}')
            if mod == 'cab':
                res['iterations'] = 100
            model_after_params_optimize[mod] = model(**res)

        return model_after_params_optimize

    def save_model(self, model, name):
        joblib.dump(model, r'{}\\{}-{}.pkl'.format(self.export_path, name, datetime.date.today()))

    def main(self):
        logger.add('{}\\prediction-runtime.txt'.format(self.export_path))
        logger.info('the process of obtaining UHPC properties prediction model start...')
        logger.warning('the total time depends on what the algorithm is selected by this framework')

        data, minBound, maxBound = self.preprocess()
        logger.info(f'the shape of data = {data.shape}')
        logger.info(f'↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\nmin_bound =\n{minBound}\nmax_bound =\n{maxBound}')
        bound_df = pd.DataFrame(np.array([minBound, maxBound]))
        bound_df.to_excel(f'{self.export_path}\\bound.xlsx')

        total = data.shape[0]
        trainval, test = data.iloc[:int(0.7 * total)], data.iloc[int(0.7 * total):]
        train_feature, train_target, val_feature, val_target = self.split_data(trainval.iloc[:, :-1],
                                                                               trainval.iloc[:, -1])
        test_feature, test_target = test.iloc[:, :-1], test.iloc[:, -1]

        algos = self.filter_model(train_feature, val_feature, train_target, val_target)
        logger.info('optimal algorithms after filter process: {}'.format(algos))

        algos_after_optimized = self.hyper_parameter(algos, train_feature, val_feature, train_target, val_target)
        logger.info('optimal algorithms after hyper_parameter optimization process: {}'.format(algos_after_optimized))

        ans = 0
        for name, algo in algos_after_optimized.items():
            algo.fit(train_feature, train_target)
            if r2_score(test_target, algo.predict(test_feature)) > ans:
                ans = r2_score(test_target, algo.predict(test_feature))
                res = (name, algo, ans)
        logger.info('the best algorithm is {}, which r2 score is {}'.format(res[0], round(res[2], 3)))
        self.save_model(res[1], res[0])
        logger.info('END!')
        mb.showinfo('condition', 'the whole progress is DONE')
