from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.core.problem import Problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator, get_reference_directions

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import re
from loguru import logger
import joblib
from tkinter import messagebox as mb


class PREPARE:
    def __init__(self, path_list, constrain_list, export_path):
        self.path_list = path_list
        self.constraint_list = constrain_list
        self.export_path = export_path

    def read_model(self):
        model_list = []
        features = 0
        for path in self.path_list:
            model = joblib.load(path)
            if hasattr(model, 'n_features_'):
                features = max(features, model.n_features_)
            elif hasattr(model, 'feature_importances_'):
                features = max(features, model.feature_importances_.shape[0])
            model_list.append(model)
        return model_list, features

    def analyse_constraint(self, line, flag):
        MV = 1e-5
        if flag == 0:
            pass
        elif flag == 1:
            part = line.split('<=')
            if len(part) == 2:
                return [(part[1] + '-' + part[0], -1)]
            elif len(part) == 3:
                return [(part[1] + '-' + part[0], -1), (part[1] + '-' + part[2], 1)]
        elif flag == 2:
            min_value, max_value = [], []
            for pairs in line.split(';'):
                pair = pairs.split(',')
                m1, m2 = float(pair[0]), float(pair[1])
                if m1 == m2:
                    min_value.append(m1)
                    max_value.append(m2 + MV)
                else:
                    min_value.append(m1)
                    max_value.append(m2)
            return [np.array(min_value), np.array(max_value)]

        elif flag == 3:
            return [line]

        elif flag == 4:
            value_list = re.split(r'\s+', line)
            value_list = list(map(lambda x: float(x), value_list))
            return np.array(value_list)

    def read_constraint(self):
        flag = 0
        Inequality = []
        MMbound = []
        Custom_function = []
        Data_bound = []
        f = open(self.constraint_list, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == 'Inequality:':
                flag = 1
            elif line == 'Bound:':
                flag = 2
            elif line == 'Customize:':
                flag = 3
            elif line == 'Standard:':
                flag = 4
            if line:
                if flag == 1 and line != 'Inequality:':
                    ans = self.analyse_constraint(line, flag)
                    Inequality.extend(ans)
                elif flag == 2 and line != 'Bound:':
                    ans = self.analyse_constraint(line, flag)
                    MMbound.extend(ans)
                elif flag == 3 and line != 'Customize:':
                    ans = self.analyse_constraint(line, flag)
                    Custom_function.extend(ans)
                elif flag == 4 and line != 'Standard:':
                    ans = self.analyse_constraint(line, flag)
                    Data_bound.append(ans)
            else:
                break
        f.close()

        return Inequality, MMbound, Custom_function, Data_bound


def get_result(res, n_obj, inverse_mm):
    hv = get_performance_indicator("hv", ref_point=np.array([1.15 for _ in range(n_obj)]))
    function_result = np.absolute(res.F)
    variable_result = np.absolute(res.X)
    variable_result = inverse_mm.transform(variable_result)
    mm = MinMaxScaler()
    targetSet = mm.fit_transform(function_result)
    logger.info(f'the value of hv indicator is {hv.do(targetSet)}')
    return function_result, variable_result


def main(model_path, constraint_path, export_path):
    logger.add(f'{export_path}//moo-runtime.txt')
    logger.info('the multi-object optimization process is in progress!')

    summary = PREPARE(model_path, constraint_path, export_path)
    model_list, num_features = summary.read_model()
    inequality, custom_bound, custom_function, data_bound = summary.read_constraint()
    logger.info('constrain information has already imported!')
    print(
        f'inequality={inequality}\ncustom_bound={custom_bound}\ncustom_function={custom_function}\ndata_bound={data_bound}')

    if len(custom_bound) != 2 or len(data_bound) != 2:
        mb.showwarning('error', 'lower and upper bound should be one item')

    MM = MinMaxScaler()
    MM.fit(np.vstack([data_bound[0], data_bound[1]]))

    n_var = num_features
    n_obj = len(model_list) + len(custom_function)
    n_constr = len(inequality)
    xl = MM.transform(custom_bound[0][np.newaxis, :]).ravel().tolist()
    xu = MM.transform(custom_bound[1][np.newaxis, :]).ravel().tolist()

    class MOOProblem(Problem):
        def __init__(self):
            super(MOOProblem, self).__init__(n_var=n_var,
                                             n_obj=n_obj,
                                             n_constr=n_constr,
                                             xl=xl,
                                             xu=xu)
            self.scaler = MM
            self.fPart = []
            self.gPart = []

        def _evaluate(self, x, out, *args, **kwargs):
            fPart, gPart = [], []

            def inverse(i):
                data_min, data_max = self.scaler.data_min_, self.scaler.data_max_
                return x[:, i] * (data_max[i] - data_min[i]) + data_min[i]

            for model in model_list:
                ans = model.predict(x)
                fPart.append(-ans)
            ans = eval(custom_function[0])
            fPart.append(ans)
            out['F'] = np.column_stack(fPart)

            for constraint in inequality:
                eq, gtrt = constraint[0], constraint[1]
                eq = eval(eq)
                if gtrt == 1:
                    gPart.append(eq)
                else:
                    gPart.append(-eq)
            out['G'] = np.column_stack(gPart)

    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    the_algorithm = CTAEA(ref_dirs=ref_dirs,
                          mutation=get_mutation("real_pm", eta=15),
                          sampling=get_sampling("real_random"),
                          crossover=get_crossover("real_sbx", eta=15, prob=0.9),
                          eliminate_duplicates=True)

    logger.info('moo process is in progress')
    res = minimize(MOOProblem(), the_algorithm, ('n_gen', 600), seed=1)
    f_res, x_res = get_result(res, n_obj, MM)
    logger.info(f'after optimization, there are {f_res.shape[0]} items in result')
    fdf, xdf = pd.DataFrame(f_res), pd.DataFrame(x_res)
    fdf.to_excel(f'{export_path}//optimized_properties_result.xlsx')
    xdf.to_excel(f'{export_path}//optimized_variation_result.xlsx')
    logger.info(f'the result file exports to {export_path}')
    logger.info(f'the multi-object optimization process is DONE!')