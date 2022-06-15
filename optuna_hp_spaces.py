# -*- coding: utf-8 -*-
''' Methods to generate hyperparameter search spaces for optuna.
    
Each takes an optuna Trial object and a set of parameter names and determines
the values to be sampled from the parameter bounds specified in the TPOT 
documentation
'''
import numpy as np
import utils as u
import random
from optuna.trial import create_trial
from optuna.distributions import (CategoricalDistribution, 
                                  UniformDistribution, 
                                  IntUniformDistribution,
                                  LogUniformDistribution)

def make_hp_space_discrete(trial, param_names):
            
    trial_params = []

    for name in param_names:
        # ***** REGRESSOR HYPERPARAMETERS *****
        # ElasticNetCV hyperparameters
        if name == 'ElasticNetCV__l1_ratio':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'ElasticNetCV__l1_ratio', 0.0, 1.0, 0.05)))
        elif name == 'ElasticNetCV__tol':
            trial_params.append((name, trial.suggest_categorical(
                'ElasticNetCV__tol', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])))
        # ExtraTreesRegressor hyperparameters
        elif name == 'ExtraTreesRegressor__n_estimators':
            # trial_params.append(trial.suggest_int(
            #     'ExtraTreesRegressor__n_estimators', 100, 100))
            trial_params.append((name,'skip'))
        elif name == 'ExtraTreesRegressor__max_features':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'ExtraTreesRegressor__max_features', 0.05, 1.0, 0.05)))
        elif name == 'ExtraTreesRegressor__min_samples_split':
            trial_params.append((name, trial.suggest_int(
               'ExtraTreesRegressor__min_samples_split', 2, 20)))
        elif name == 'ExtraTreesRegressor__min_samples_leaf':
                trial_params.append((name, trial.suggest_int(
                   'ExtraTreesRegressor__min_samples_leaf', 1, 20)))
        elif name == 'ExtraTreesRegressor__bootstrap':
                trial_params.append((name, trial.suggest_categorical(
                   'ExtraTreesRegressor__bootstrap', ['True', 'False'])))
        # GradientBoostingRegressor hyperparameters
        elif name == 'GradientBoostingRegressor__n_estimators':
            # trial_params.append(trial.suggest_int(
            #    'GradientBoostingRegressor__n_estimators', 100, 100))
            trial_params.append((name, 'skip'))
        elif name == 'GradientBoostingRegressor__loss':
            trial_params.append((name, trial.suggest_categorical(
               'GradientBoostingRegressor__loss', 
               ['ls', 'lad', 'huber', 'quantile'])))
        elif name == 'GradientBoostingRegressor__learning_rate':
            trial_params.append((name, trial.suggest_categorical(
                'GradientBoostingRegressor__learning_rate', 
                [1e-3, 1e-2, 1e-1, 0.5, 1.])))
        elif name == 'GradientBoostingRegressor__max_depth':
            trial_params.append((name, trial.suggest_int(
                'GradientBoostingRegressor__max_depth', 1, 10)))
        elif name == 'GradientBoostingRegressor__min_samples_split':
            trial_params.append((name, trial.suggest_int(
                'GradientBoostingRegressor__min_samples_split', 2, 20)))
        elif name == 'GradientBoostingRegressor__min_samples_leaf':
            trial_params.append((name, trial.suggest_int(
               'GradientBoostingRegressor__min_samples_leaf', 1, 20)))
        elif name == 'GradientBoostingRegressor__subsample':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'GradientBoostingRegressor__subsample', 0.05, 1.0, 0.05)))
        elif name == 'GradientBoostingRegressor__max_features':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'GradientBoostingRegressor__max_features', 
                0.05, 1.0, 0.05)))
        elif name == 'GradientBoostingRegressor__alpha':
            trial_params.append((name, trial.suggest_categorical(
               'GradientBoostingRegressor__alpha', 
               [0.75, 0.8, 0.85, 0.9, 0.95, 0.99])))
        # AdaBoostRegressor hyperparameters
        elif name == 'AdaBoostRegressor__n_estimators':
            # trial_params.append(trial.suggest_int(
            #     'AdaBoostRegressor__n_estimators', 100, 100))
            trial_params.append((name, 'skip'))
        elif name == 'AdaBoostRegressor__learning_rate':
            trial_params.append((name, trial.suggest_categorical(
              'AdaBoostRegressor__learning_rate', 
              [1e-3, 1e-2, 1e-1, 0.5, 1.])))
        elif name == 'AdaBoostRegressor__loss':
            trial_params.append((name, trial.suggest_categorical(
              'AdaBoostRegressor__loss', 
              ["linear", "square", "exponential"])))
        # DecisionTreeRegressor hyperparameters
        elif name == 'DecisionTreeRegressor__max_depth':
            trial_params.append((name, trial.suggest_int(
               'DecisionTreeRegressor__max_depth', 1, 10)))
        elif name == 'DecisionTreeRegressor__min_samples_split':
            trial_params.append((name, trial.suggest_int(
               'DecisionTreeRegressor__min_samples_split', 2, 20)))
        elif name == 'DecisionTreeRegressor__min_samples_leaf':
            trial_params.append((name, trial.suggest_int(
                'DecisionTreeRegressor__min_samples_leaf', 1, 20)))
        # KNeighborsRegressor hyperparameters
        elif name == 'KNeighborsRegressor__n_neighbors':
            trial_params.append((name, trial.suggest_int(
               'KNeighborsRegressor__n_neighbors', 1, 100)))
        elif name == 'KNeighborsRegressor__weights':
            trial_params.append((name, trial.suggest_categorical(
               'KNeighborsRegressor__weights', ["uniform", "distance"])))
        elif name == 'KNeighborsRegressor__p':
            trial_params.append((name, trial.suggest_int(
                'KNeighborsRegressor__p', 1, 2)))
        # LassoLarsCV hyperparameters
        elif name == 'LassoLarsCV__normalize':
            trial_params.append((name, trial.suggest_categorical(
               'LassoLarsCV__normalize', ['True','False'])))
        # LinearSVR hyperparameters
        elif name == 'LinearSVR__loss':
            trial_params.append((name, trial.suggest_categorical(
               'LinearSVR__loss', 
               ["epsilon_insensitive", "squared_epsilon_insensitive"])))
        elif name == 'LinearSVR__dual':
            trial_params.append((name, trial.suggest_categorical(
                'LinearSVR__dual', ['True', 'False'])))
        elif name == 'LinearSVR__tol':
            trial_params.append((name, trial.suggest_categorical(
                'LinearSVR__tol', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])))
        elif name == 'LinearSVR__C':
            trial_params.append((name, trial.suggest_categorical(
                'LinearSVR__C', 
                [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.])))
        elif name == 'LinearSVR__epsilon':
            trial_params.append((name, trial.suggest_categorical(
                'LinearSVR__epsilon', [1e-4, 1e-3, 1e-2, 1e-1, 1.])))
        # RandomForestRegressor hyperparameters
        elif name == 'RandomForestRegressor__bootstrap':
            trial_params.append((name, trial.suggest_categorical(
                'RandomForestRegressor__bootstrap',['True','False'])))
        elif name == 'RandomForestRegressor__max_features':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'RandomForestRegressor__max_features', 0.05, 1.0, 0.05)))
        elif name == 'RandomForestRegressor__min_samples_leaf':
            trial_params.append((name, trial.suggest_int(
                'RandomForestRegressor__min_samples_leaf', 1, 20)))
        elif name == 'RandomForestRegressor__min_samples_split':
            trial_params.append((name, trial.suggest_int(
                'RandomForestRegressor__min_samples_split', 2, 20)))
        elif name == 'RandomForestRegressor__n_estimators':
            # trial_params.append((name, trial.suggest_int(
            #     'RandomForestRegressor__n_estimators', 100, 100))
            trial_params.append((name, 'skip'))
        # XGBRegressor hyperparameters
        elif name == 'XGBRegressor__learning_rate':
            trial_params.append((name, trial.suggest_categorical(
                'XGBRegressor__learning_rate', [1e-3, 1e-2, 1e-1, 0.5, 1.])))
        elif name == 'XGBRegressor__max_depth':
            trial_params.append((name, trial.suggest_int(
                'XGBRegressor__max_depth', 1, 10)))
        elif name == 'XGBRegressor__min_child_weight':
            trial_params.append((name, trial.suggest_int(
                'XGBRegressor__min_child_weight', 1, 20)))
        elif name == 'XGBRegressor__n_estimators':
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__n_estimators', 100, 100))
            trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__n_jobs':
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__n_jobs', 1, 1)))
            trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__objective':
            # trial_params.append((name, trial.suggest_categorical(
            #     'XGBRegressor__objective', ['reg:squarederror'])))
            trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__subsample':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'XGBRegressor__subsample', 0.05, 1.0, 0.05)))
        elif name == 'XGBRegressor__verbosity':
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__verbosity', 0, 0)))
            trial_params.append((name, 'skip'))
        # SGDRegressor hyperparameters
        elif name == 'SGDRegressor__loss':
            trial_params.append((name, trial.suggest_categorical(
               'SGDRegressor__loss', 
               ['squared_loss', 'huber', 'epsilon_insensitive'])))
        elif name == 'SGDRegressor__penalty':
            # trial_params.append((name, trial.suggest_categorical(
            #     'SGDRegressor__penalty', ['elasticnet'])))
            trial_params.append((name, 'skip'))
        elif name == 'SGDRegressor__alpha':
            trial_params.append((name, trial.suggest_categorical(
                'SGDRegressor__alpha', [0.0, 0.01, 0.001])))
        elif name == 'SGDRegressor__learning_rate':
            trial_params.append((name, trial.suggest_categorical(
                'SGDRegressor__learning_rate', ['invscaling', 'constant'])))
        elif name == 'SGDRegressor__fit_intercept':
            trial_params.append((name, trial.suggest_categorical(
                'SGDRegressor__fit_intercept', ['True', 'False'])))
        elif name == 'SGDRegressor__l1_ratio':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'SGDRegressor__l1_ratio', 0.0, 1.0, 0.25)))
        elif name == 'SGDRegressor__eta0':
            trial_params.append((name, trial.suggest_categorical(
                'SGDRegressor__eta0', [0.1, 1.0, 0.01])))
        elif name == 'SGDRegressor__power_t':
            trial_params.append((name, trial.suggest_categorical(
                'SGDRegressor__power_t',
                [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0])))
        # ***** PREPROCESSOR HYPERPARAMETERS *****
        # Binarizer hyperparameters
        elif name == 'Binarizer__threshold':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'Binarizer__threshold', 0.0, 1.0, 0.05)))
        # FastICA hyperparameters
        elif name == 'FastICA__tol':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'FastICA__tol', 0.0, 1.0, 0.05)))
        # FeatureAgglomeration hyperparameters
        elif name == 'FeatureAgglomeration__linkage':
            trial_params.append((name, trial.suggest_categorical(
                'FeatureAgglomeration__linkage', 
                ['ward', 'complete', 'average'])))
        elif name == 'FeatureAgglomeration__affinity':
            trial_params.append((name, trial.suggest_categorical(
                'FeatureAgglomeration__affinity', 
                ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'])))
        # Normalizer hyperparameters
        elif name == 'Normalizer__norm':
            trial_params.append((name, trial.suggest_categorical(
                'Normalizer__norm', ['l1', 'l2', 'max'])))
        # Nystroem hyperparameters
        elif name == 'Nystroem__kernel':
            trial_params.append((name, trial.suggest_categorical(
                'Nystroem__kernel', 
                ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 
                 'poly', 'linear', 'additive_chi2', 'sigmoid'])))
        elif name == 'Nystroem__gamma':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'Nystroem__gamma', 0.0, 1.0, 0.05)))
        elif name == 'Nystroem__n_components':
            trial_params.append((name, trial.suggest_int(
                'Nystroem__n_components', 1, 10)))
        # PCA hyperparameters
        elif name == 'PCA__svd_solver':
            # trial_params.append((name, trial.suggest_categorical(
            #     'PCA__svd_solver', ['randomized'])))
            trial_params.append((name, 'skip'))
        elif name == 'PCA__iterated_power':
            trial_params.append((name, trial.suggest_int(
                'PCA__iterated_power', 1, 10)))
        # PolynomialFeatures hyperparameters
        elif name == 'PolynomialFeatures__degree':
            # trial_params.append((name, trial.suggest_int(
            #     'PolynomialFeatures__degree', 2, 2)))
            trial_params.append((name, 'skip'))
        elif name == 'PolynomialFeatures__include_bias':
            # trial_params.append((name, trial.suggest_categorical(
            #     'PolynomialFeatures__include_bias', ['False'])))
            trial_params.append((name, 'skip'))
        elif name == 'PolynomialFeatures__interaction_only':
            # trial_params.append((name, trial.suggest_categorical(
            #     'PolynomialFeatures__interaction_only', ['False'])))
            trial_params.append((name, 'skip'))
        # RBFSampler hyperparameters
        elif name == 'RBFSampler__gamma':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'RBFSampler__gamma', 0.0, 1.0, 0.05)))
        # OneHotEncoder hyperparameters
        elif name == 'OneHotEncoder__minimum_fraction':
            trial_params.append((name, trial.suggest_categorical(
                'OneHotEncoder__minimum_fraction', 
                [0.05, 0.1, 0.15, 0.2, 0.25])))
        elif name == 'OneHotEncoder__sparse':
            # trial_params.append((name, trial.suggest_categorical(
            #     'OneHotEncoder__sparse', ['False'])))
            trial_params.append((name, 'skip'))
        elif name == 'OneHotEncoder__threshold':
            # trial_params.append((name, trial.suggest_int(
            #     'OneHotEncoder__threshold', 10, 10)))
            trial_params.append((name, 'skip'))
        # ***** SELECTOR HYPERPARAMETERS *****
        # SelectFwe hyperparameters
        elif name == 'SelectFwe__alpha':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'SelectFwe__alpha', 0, 0.05, 0.001)))
        # SelectPercentile hyperparameters
        elif name == 'SelectPercentile__percentile':
            trial_params.append((name, trial.suggest_int(
                'SelectPercentile_percentile', 1, 99)))
        # VarianceThreshold hyperparameters
        elif name == 'VarianceThreshold__threshold':
            trial_params.append((name, trial.suggest_categorical(
                'VarianceThreshold__threshold', 
                [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2])))
        # SelectFromModel hyperparameters
        elif name == 'SelectFromModel__threshold':
            trial_params.append((name, trial.suggest_discrete_uniform(
                'SelectFromModel__threshold', 0.0, 1.0, 0.05)))
        elif name == 'SelectFromModel__ExtraTreesRegressor__n_estimators':
            trial_params.append((name, 'skip'))
        elif name == 'SelectFromModel__ExtraTreesRegressor__max_features':
            trial_params.append((name, trial.suggest_float(
                'SelectFromModel__ExtraTreesRegressor__max_features', 0.05, 1.0)))
        else:
            print("Unable to parse parameter name: " 
                  + name + ", skipping..")
            trial_params.append((name, 'skip'))
                
    return trial_params

def make_hp_space_real(trial, param_names):
            
    trial_params = []

    for name in param_names:
        # ***** REGRESSOR HYPERPARAMETERS *****
        # ElasticNetCV hyperparameters
        if name == 'ElasticNetCV__l1_ratio':
            trial_params.append((name, trial.suggest_float(
                'ElasticNetCV__l1_ratio', 0.0, 1.0)))
        elif name == 'ElasticNetCV__tol':
            trial_params.append((name, trial.suggest_float(
                'ElasticNetCV__tol', 1e-5, 1e-1, log=True)))
        # ExtraTreesRegressor hyperparameters
        elif name == 'ExtraTreesRegressor__n_estimators':
            # trial_params.append(trial.suggest_int(
            #     'ExtraTreesRegressor__n_estimators', 100, 100))
            trial_params.append((name,'skip'))
        elif name == 'ExtraTreesRegressor__max_features':
            trial_params.append((name, trial.suggest_float(
                'ExtraTreesRegressor__max_features', 0.05, 1.0)))
        elif name == 'ExtraTreesRegressor__min_samples_split':
            trial_params.append((name, trial.suggest_int(
               'ExtraTreesRegressor__min_samples_split', 2, 20)))
        elif name == 'ExtraTreesRegressor__min_samples_leaf':
            trial_params.append((name, trial.suggest_int(
                   'ExtraTreesRegressor__min_samples_leaf', 1, 20)))
        elif name == 'ExtraTreesRegressor__bootstrap':
            trial_params.append((name, trial.suggest_categorical(
                   'ExtraTreesRegressor__bootstrap', ['True', 'False'])))
        # GradientBoostingRegressor hyperparameters
        elif name == 'GradientBoostingRegressor__n_estimators':
            # trial_params.append(trial.suggest_int(
            #    'GradientBoostingRegressor__n_estimators', 100, 100))
            trial_params.append((name, 'skip'))
        elif name == 'GradientBoostingRegressor__loss':
            trial_params.append((name, trial.suggest_categorical(
               'GradientBoostingRegressor__loss', 
               ['ls', 'lad', 'huber', 'quantile'])))
        elif name == 'GradientBoostingRegressor__learning_rate':
            trial_params.append((name, trial.suggest_float(
                'GradientBoostingRegressor__learning_rate', 
                1e-3, 1., log=True)))
        elif name == 'GradientBoostingRegressor__max_depth':
            trial_params.append((name, trial.suggest_int(
                'GradientBoostingRegressor__max_depth', 1, 10)))
        elif name == 'GradientBoostingRegressor__min_samples_split':
            trial_params.append((name, trial.suggest_int(
                'GradientBoostingRegressor__min_samples_split', 2, 20)))
        elif name == 'GradientBoostingRegressor__min_samples_leaf':
            trial_params.append((name, trial.suggest_int(
               'GradientBoostingRegressor__min_samples_leaf', 1, 20)))
        elif name == 'GradientBoostingRegressor__subsample':
            trial_params.append((name, trial.suggest_float(
                'GradientBoostingRegressor__subsample', 0.05, 1.0)))
        elif name == 'GradientBoostingRegressor__max_features':
            trial_params.append((name, trial.suggest_float(
                'GradientBoostingRegressor__max_features', 
                0.05, 1.0)))
        elif name == 'GradientBoostingRegressor__alpha':
            trial_params.append((name, trial.suggest_float(
               'GradientBoostingRegressor__alpha', 0.75, 0.99)))
        # AdaBoostRegressor hyperparameters
        elif name == 'AdaBoostRegressor__n_estimators':
            # trial_params.append(trial.suggest_int(
            #     'AdaBoostRegressor__n_estimators', 100, 100))
            trial_params.append((name, 'skip'))
        elif name == 'AdaBoostRegressor__learning_rate':
            trial_params.append((name, trial.suggest_float(
              'AdaBoostRegressor__learning_rate', 1e-3, 1., log=True)))
        elif name == 'AdaBoostRegressor__loss':
            trial_params.append((name, trial.suggest_categorical(
              'AdaBoostRegressor__loss', 
              ["linear", "square", "exponential"])))
        # DecisionTreeRegressor hyperparameters
        elif name == 'DecisionTreeRegressor__max_depth':
            trial_params.append((name, trial.suggest_int(
               'DecisionTreeRegressor__max_depth', 1, 10)))
        elif name == 'DecisionTreeRegressor__min_samples_split':
            trial_params.append((name, trial.suggest_int(
               'DecisionTreeRegressor__min_samples_split', 2, 20)))
        elif name == 'DecisionTreeRegressor__min_samples_leaf':
            trial_params.append((name, trial.suggest_int(
                'DecisionTreeRegressor__min_samples_leaf', 1, 20)))
        # KNeighborsRegressor hyperparameters
        elif name == 'KNeighborsRegressor__n_neighbors':
            trial_params.append((name, trial.suggest_int(
               'KNeighborsRegressor__n_neighbors', 1, 100)))
        elif name == 'KNeighborsRegressor__weights':
            trial_params.append((name, trial.suggest_categorical(
               'KNeighborsRegressor__weights', ["uniform", "distance"])))
        elif name == 'KNeighborsRegressor__p':
            trial_params.append((name, trial.suggest_int(
                'KNeighborsRegressor__p', 1, 2)))
        # LassoLarsCV hyperparameters
        elif name == 'LassoLarsCV__normalize':
            trial_params.append((name, trial.suggest_categorical(
               'LassoLarsCV__normalize', ['True','False'])))
        # LinearSVR hyperparameters
        elif name == 'LinearSVR__loss':
            trial_params.append((name, trial.suggest_categorical(
               'LinearSVR__loss', 
               ["epsilon_insensitive", "squared_epsilon_insensitive"])))
        elif name == 'LinearSVR__dual':
            trial_params.append((name, trial.suggest_categorical(
                'LinearSVR__dual', ['True', 'False'])))
        elif name == 'LinearSVR__tol':
            trial_params.append((name, trial.suggest_float(
                'LinearSVR__tol', 1e-5, 1e-1, log=True)))
        elif name == 'LinearSVR__C':
            trial_params.append((name, trial.suggest_float(
                'LinearSVR__C', 1e-4, 25., log=True)))
        elif name == 'LinearSVR__epsilon':
            trial_params.append((name, trial.suggest_float(
                'LinearSVR__epsilon', 1e-4, 1., log=True)))
        # RandomForestRegressor hyperparameters
        elif name == 'RandomForestRegressor__bootstrap':
            trial_params.append((name, trial.suggest_categorical(
                'RandomForestRegressor__bootstrap',['True','False'])))
        elif name == 'RandomForestRegressor__max_features':
            trial_params.append((name, trial.suggest_float(
                'RandomForestRegressor__max_features', 0.05, 1.0)))
        elif name == 'RandomForestRegressor__min_samples_leaf':
            trial_params.append((name, trial.suggest_int(
                'RandomForestRegressor__min_samples_leaf', 1, 20)))
        elif name == 'RandomForestRegressor__min_samples_split':
            trial_params.append((name, trial.suggest_int(
                'RandomForestRegressor__min_samples_split', 2, 20)))
        elif name == 'RandomForestRegressor__n_estimators':
            # trial_params.append((name, trial.suggest_int(
            #     'RandomForestRegressor__n_estimators', 100, 100))
            trial_params.append((name, 'skip'))
        # XGBRegressor hyperparameters
        elif name == 'XGBRegressor__learning_rate':
            trial_params.append((name, trial.suggest_float(
                'XGBRegressor__learning_rate', 1e-3, 1., log=True)))
        elif name == 'XGBRegressor__max_depth':
            trial_params.append((name, trial.suggest_int(
                'XGBRegressor__max_depth', 1, 10)))
        elif name == 'XGBRegressor__min_child_weight':
            trial_params.append((name, trial.suggest_int(
                'XGBRegressor__min_child_weight', 1, 20)))
        elif name == 'XGBRegressor__n_estimators':
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__n_estimators', 100, 100))
            trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__n_jobs':
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__n_jobs', 1, 1)))
            trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__objective':
            # trial_params.append((name, trial.suggest_categorical(
            #     'XGBRegressor__objective', ['reg:squarederror'])))
            trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__subsample':
            trial_params.append((name, trial.suggest_float(
                'XGBRegressor__subsample', 0.05, 1.0)))
        elif name == 'XGBRegressor__verbosity':
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__verbosity', 0, 0)))
            trial_params.append((name, 'skip'))
        # SGDRegressor hyperparameters
        elif name == 'SGDRegressor__loss':
            trial_params.append((name, trial.suggest_categorical(
               'SGDRegressor__loss', 
               ['squared_loss', 'huber', 'epsilon_insensitive'])))
        elif name == 'SGDRegressor__penalty':
            # trial_params.append((name, trial.suggest_categorical(
            #     'SGDRegressor__penalty', ['elasticnet'])))
            trial_params.append((name, 'skip'))
        elif name == 'SGDRegressor__alpha':
            # trial_params.append((name, trial.suggest_float(
            #     'SGDRegressor__alpha', 1e-5, 0.01, log=True)))
            trial_params.append((name, trial.suggest_float(
                'SGDRegressor__alpha', 0, 0.01)))
        elif name == 'SGDRegressor__learning_rate':
            trial_params.append((name, trial.suggest_categorical(
                'SGDRegressor__learning_rate', ['invscaling', 'constant'])))
        elif name == 'SGDRegressor__fit_intercept':
            trial_params.append((name, trial.suggest_categorical(
                'SGDRegressor__fit_intercept', ['True', 'False'])))
        elif name == 'SGDRegressor__l1_ratio':
            trial_params.append((name, trial.suggest_float(
                'SGDRegressor__l1_ratio', 0.0, 1.0)))
        elif name == 'SGDRegressor__eta0':
            trial_params.append((name, trial.suggest_float(
                'SGDRegressor__eta0', 0.01, 1.0, log=True)))
        elif name == 'SGDRegressor__power_t':
            # trial_params.append((name, trial.suggest_float(
            #     'SGDRegressor__power_t', 1e-5, 100.0, log=True)))
            trial_params.append((name, trial.suggest_float(
                'SGDRegressor__power_t', 0, 100.0)))
        # ***** PREPROCESSOR HYPERPARAMETERS *****
        # Binarizer hyperparameters
        elif name == 'Binarizer__threshold':
            trial_params.append((name, trial.suggest_float(
                'Binarizer__threshold', 0.0, 1.0)))
        # FastICA hyperparameters
        elif name == 'FastICA__tol':
            trial_params.append((name, trial.suggest_float(
                'FastICA__tol', 0.0, 1.0)))
        # FeatureAgglomeration hyperparameters
        elif name == 'FeatureAgglomeration__linkage':
            trial_params.append((name, trial.suggest_categorical(
                'FeatureAgglomeration__linkage', 
                ['ward', 'complete', 'average'])))
        elif name == 'FeatureAgglomeration__affinity':
            trial_params.append((name, trial.suggest_categorical(
                'FeatureAgglomeration__affinity', 
                ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'])))
        # Normalizer hyperparameters
        elif name == 'Normalizer__norm':
            trial_params.append((name, trial.suggest_categorical(
                'Normalizer__norm', ['l1', 'l2', 'max'])))
        # Nystroem hyperparameters
        elif name == 'Nystroem__kernel':
            trial_params.append((name, trial.suggest_categorical(
                'Nystroem__kernel', 
                ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 
                 'poly', 'linear', 'additive_chi2', 'sigmoid'])))
        elif name == 'Nystroem__gamma':
            trial_params.append((name, trial.suggest_float(
                'Nystroem__gamma', 0.0, 1.0)))
        elif name == 'Nystroem__n_components':
            trial_params.append((name, trial.suggest_int(
                'Nystroem__n_components', 1, 10)))
        # PCA hyperparameters
        elif name == 'PCA__svd_solver':
            # trial_params.append((name, trial.suggest_categorical(
            #     'PCA__svd_solver', ['randomized'])))
            trial_params.append((name, 'skip'))
        elif name == 'PCA__iterated_power':
            trial_params.append((name, trial.suggest_int(
                'PCA__iterated_power', 1, 10)))
        # PolynomialFeatures hyperparameters
        elif name == 'PolynomialFeatures__degree':
            # trial_params.append((name, trial.suggest_int(
            #     'PolynomialFeatures__degree', 2, 2)))
            trial_params.append((name, 'skip'))
        elif name == 'PolynomialFeatures__include_bias':
            # trial_params.append((name, trial.suggest_categorical(
            #     'PolynomialFeatures__include_bias', ['False'])))
            trial_params.append((name, 'skip'))
        elif name == 'PolynomialFeatures__interaction_only':
            # trial_params.append((name, trial.suggest_categorical(
            #     'PolynomialFeatures__interaction_only', ['False'])))
            trial_params.append((name, 'skip'))
        # RBFSampler hyperparameters
        elif name == 'RBFSampler__gamma':
            trial_params.append((name, trial.suggest_float(
                'RBFSampler__gamma', 0.0, 1.0)))
        # OneHotEncoder hyperparameters
        elif name == 'OneHotEncoder__minimum_fraction':
            trial_params.append((name, trial.suggest_float(
                'OneHotEncoder__minimum_fraction', 0.05, 0.25)))
        elif name == 'OneHotEncoder__sparse':
            # trial_params.append((name, trial.suggest_categorical(
            #     'OneHotEncoder__sparse', ['False'])))
            trial_params.append((name, 'skip'))
        elif name == 'OneHotEncoder__threshold':
            # trial_params.append((name, trial.suggest_int(
            #     'OneHotEncoder__threshold', 10, 10)))
            trial_params.append((name, 'skip'))
        # ***** SELECTOR HYPERPARAMETERS *****
        # SelectFwe hyperparameters
        elif name == 'SelectFwe__alpha':
            trial_params.append((name, trial.suggest_float(
                'SelectFwe__alpha', 0, 0.05)))
        # SelectPercentile hyperparameters
        elif name == 'SelectPercentile__percentile':
            trial_params.append((name, trial.suggest_int(
                'SelectPercentile_percentile', 1, 99)))
        # VarianceThreshold hyperparameters
        elif name == 'VarianceThreshold__threshold':
            trial_params.append((name, trial.suggest_float(
                'VarianceThreshold__threshold', 0.0001, 0.2, log=True)))
        # SelectFwe hyperparameters
        elif name == 'SelectFromModel__threshold':
            # trial_params.append((name, 'skip'))
            trial_params.append((name, trial.suggest_float(
                'SelectFromModel__threshold', 0, 1.0)))
        elif name == 'SelectFromModel__ExtraTreesRegressor__n_estimators':
            trial_params.append((name, 'skip'))
        elif name == 'SelectFromModel__ExtraTreesRegressor__max_features':
            # trial_params.append((name, 'skip'))
            trial_params.append((name, trial.suggest_float(
                'SelectFromModel__ExtraTreesRegressor__max_features', 0.05, 1.0)))
        else:
            print("Unable to parse parameter name: " 
                  + name + ", skipping..")
            trial_params.append((name, 'skip'))
                
    return trial_params


def make_optuna_trial(trial_params, value):
    
    params = {}
    distributions = {}
    
    for (name,val) in trial_params:
        # ***** REGRESSOR HYPERPARAMETERS *****
        # ElasticNetCV hyperparameters
        if name == 'ElasticNetCV__l1_ratio':
            distributions[name] = UniformDistribution(0.0, 1.0)
            params[name] = val
        elif name == 'ElasticNetCV__tol':
            distributions[name] = LogUniformDistribution(1e-5, 1e-1)
            params[name] = val
        # ExtraTreesRegressor hyperparameters
        elif name == 'ExtraTreesRegressor__n_estimators':
            continue
            # trial_params.append(trial.suggest_int(
            #     'ExtraTreesRegressor__n_estimators', 100, 100))
            # trial_params.append((name,'skip'))
        elif name == 'ExtraTreesRegressor__max_features':
            distributions[name] = UniformDistribution(0.05, 1.0)
            params[name] = val
        elif name == 'ExtraTreesRegressor__min_samples_split':
            distributions[name] = IntUniformDistribution(2, 20)
            params[name] = val
        elif name == 'ExtraTreesRegressor__min_samples_leaf':
            distributions[name] = IntUniformDistribution(1, 20)
            params[name] = val
        elif name == 'ExtraTreesRegressor__bootstrap':
            distributions[name] = CategoricalDistribution(['True', 'False'])
            params[name] = val
        # GradientBoostingRegressor hyperparameters
        elif name == 'GradientBoostingRegressor__n_estimators':
            continue
            # trial_params.append(trial.suggest_int(
            #    'GradientBoostingRegressor__n_estimators', 100, 100))
            # trial_params.append((name, 'skip'))
        elif name == 'GradientBoostingRegressor__loss':
            distributions[name] = CategoricalDistribution(
                ['ls', 'lad', 'huber', 'quantile'])
            params[name] = val
        elif name == 'GradientBoostingRegressor__learning_rate':
            distributions[name] = LogUniformDistribution(1e-3, 1.)
            params[name] = val
        elif name == 'GradientBoostingRegressor__max_depth':
            distributions[name] = IntUniformDistribution(1, 10)
            params[name] = val
        elif name == 'GradientBoostingRegressor__min_samples_split':
            distributions[name] = IntUniformDistribution(2, 20)
            params[name] = val
        elif name == 'GradientBoostingRegressor__min_samples_leaf':
            distributions[name] = IntUniformDistribution(1, 20)
            params[name] = val
        elif name == 'GradientBoostingRegressor__subsample':
            distributions[name] = UniformDistribution(0.05, 1.0)
            params[name] = val
        elif name == 'GradientBoostingRegressor__max_features':
            distributions[name] = UniformDistribution(0.05, 1.0)
            params[name] = val
        elif name == 'GradientBoostingRegressor__alpha':
            distributions[name] = UniformDistribution(0.75, 0.99)
            params[name] = val
        # AdaBoostRegressor hyperparameters
        elif name == 'AdaBoostRegressor__n_estimators':
            continue
            # trial_params.append(trial.suggest_int(
            #     'AdaBoostRegressor__n_estimators', 100, 100))
            # trial_params.append((name, 'skip'))
        elif name == 'AdaBoostRegressor__learning_rate':
            distributions[name] = LogUniformDistribution(1e-3, 1.)
            params[name] = val
        elif name == 'AdaBoostRegressor__loss':
            distributions[name] = CategoricalDistribution(
                ["linear", "square", "exponential"])
            params[name] = val
        # DecisionTreeRegressor hyperparameters
        elif name == 'DecisionTreeRegressor__max_depth':
            distributions[name] = IntUniformDistribution(1, 10)
            params[name] = val
        elif name == 'DecisionTreeRegressor__min_samples_split':
            distributions[name] = IntUniformDistribution(2, 20)
            params[name] = val
        elif name == 'DecisionTreeRegressor__min_samples_leaf':
            distributions[name] = IntUniformDistribution(1, 20)
            params[name] = val
        # KNeighborsRegressor hyperparameters
        elif name == 'KNeighborsRegressor__n_neighbors':
            distributions[name] = IntUniformDistribution(1, 100)
            params[name] = val
        elif name == 'KNeighborsRegressor__weights':
            distributions[name] = CategoricalDistribution(
                ["uniform", "distance"])
            params[name] = val
        elif name == 'KNeighborsRegressor__p':
            distributions[name] = IntUniformDistribution(1, 2)
            params[name] = val
        # LassoLarsCV hyperparameters
        elif name == 'LassoLarsCV__normalize':
            distributions[name] = CategoricalDistribution(['True','False'])
            params[name] = val
        # LinearSVR hyperparameters
        elif name == 'LinearSVR__loss':
            distributions[name] = CategoricalDistribution(
                ["epsilon_insensitive", "squared_epsilon_insensitive"])
            params[name] = val
        elif name == 'LinearSVR__dual':
            distributions[name] = CategoricalDistribution(['True', 'False'])
            params[name] = val
        elif name == 'LinearSVR__tol':
            distributions[name] = LogUniformDistribution(1e-5, 1e-1)
            params[name] = val
        elif name == 'LinearSVR__C':
            distributions[name] = LogUniformDistribution(1e-4, 25.)
            params[name] = val
        elif name == 'LinearSVR__epsilon':
            distributions[name] = LogUniformDistribution(1e-4, 1.)
            params[name] = val
        # RandomForestRegressor hyperparameters
        elif name == 'RandomForestRegressor__bootstrap':
            distributions[name] = CategoricalDistribution(['True','False'])
            params[name] = val
        elif name == 'RandomForestRegressor__max_features':
            distributions[name] = UniformDistribution(0.05, 1.0)
            params[name] = val
        elif name == 'RandomForestRegressor__min_samples_leaf':
            distributions[name] = IntUniformDistribution(1, 20)
            params[name] = val
        elif name == 'RandomForestRegressor__min_samples_split':
            distributions[name] = IntUniformDistribution(2, 20)
            params[name] = val
        elif name == 'RandomForestRegressor__n_estimators':
            continue
            # trial_params.append((name, trial.suggest_int(
            #     'RandomForestRegressor__n_estimators', 100, 100))
            # trial_params.append((name, 'skip'))
        # XGBRegressor hyperparameters
        elif name == 'XGBRegressor__learning_rate':
            distributions[name] = LogUniformDistribution(1e-3, 1.)
            params[name] = val
        elif name == 'XGBRegressor__max_depth':
            distributions[name] = IntUniformDistribution(1, 10)
            params[name] = val
        elif name == 'XGBRegressor__min_child_weight':
            distributions[name] = IntUniformDistribution(1, 20)
            params[name] = val
        elif name == 'XGBRegressor__n_estimators':
            continue
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__n_estimators', 100, 100))
            # trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__n_jobs':
            continue
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__n_jobs', 1, 1)))
            # trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__objective':
            continue
            # trial_params.append((name, trial.suggest_categorical(
            #     'XGBRegressor__objective', ['reg:squarederror'])))
            # trial_params.append((name, 'skip'))
        elif name == 'XGBRegressor__subsample':
            distributions[name] = UniformDistribution(0.05, 1.0)
            params[name] = val
        elif name == 'XGBRegressor__verbosity':
            continue
            # trial_params.append((name, trial.suggest_int(
            #     'XGBRegressor__verbosity', 0, 0)))
            # trial_params.append((name, 'skip'))
        # SGDRegressor hyperparameters
        elif name == 'SGDRegressor__loss':
            distributions[name] = CategoricalDistribution(
                ['squared_loss', 'huber', 'epsilon_insensitive'])
            params[name] = val
        elif name == 'SGDRegressor__penalty':
            continue
            # trial_params.append((name, trial.suggest_categorical(
            #     'SGDRegressor__penalty', ['elasticnet'])))
            # trial_params.append((name, 'skip'))
        elif name == 'SGDRegressor__alpha':
            # trial_params.append((name, trial.suggest_float(
            #     'SGDRegressor__alpha', 1e-5, 0.01, log=True)))
            distributions[name] = UniformDistribution(0, 0.01)
            params[name] = val
        elif name == 'SGDRegressor__learning_rate':
            distributions[name] = CategoricalDistribution(
                ['invscaling', 'constant'])
            params[name] = val
        elif name == 'SGDRegressor__fit_intercept':
            distributions[name] = CategoricalDistribution(['True', 'False'])
            params[name] = val
        elif name == 'SGDRegressor__l1_ratio':
            distributions[name] = UniformDistribution(0.0, 1.0)
            params[name] = val
        elif name == 'SGDRegressor__eta0':
            distributions[name] = LogUniformDistribution(0.01, 1.0)
            params[name] = val
        elif name == 'SGDRegressor__power_t':
            # trial_params.append((name, trial.suggest_float(
            #     'SGDRegressor__power_t', 1e-5, 100.0, log=True)))
            distributions[name] = UniformDistribution(0, 100.0)
            params[name] = val
        # ***** PREPROCESSOR HYPERPARAMETERS *****
        # Binarizer hyperparameters
        elif name == 'Binarizer__threshold':
            distributions[name] = UniformDistribution(0.0, 1.0)
            params[name] = val
        # FastICA hyperparameters
        elif name == 'FastICA__tol':
            distributions[name] = UniformDistribution(0.0, 1.0)
            params[name] = val
        # FeatureAgglomeration hyperparameters
        elif name == 'FeatureAgglomeration__linkage':
            distributions[name] = CategoricalDistribution(
                ['ward', 'complete', 'average'])
            params[name] = val
        elif name == 'FeatureAgglomeration__affinity':
            distributions[name] = CategoricalDistribution(
                ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'])
            params[name] = val
        # Normalizer hyperparameters
        elif name == 'Normalizer__norm':
            distributions[name] = CategoricalDistribution(['l1', 'l2', 'max'])
            params[name] = val
        # Nystroem hyperparameters
        elif name == 'Nystroem__kernel':
            distributions[name] = CategoricalDistribution(
                ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 
                 'poly', 'linear', 'additive_chi2', 'sigmoid'])
            params[name] = val
        elif name == 'Nystroem__gamma':
            distributions[name] = UniformDistribution(0.0, 1.0)
            params[name] = val
        elif name == 'Nystroem__n_components':
            distributions[name] = IntUniformDistribution(1, 10)
            params[name] = val
        # PCA hyperparameters
        elif name == 'PCA__svd_solver':
            continue
            # trial_params.append((name, trial.suggest_categorical(
            #     'PCA__svd_solver', ['randomized'])))
            # trial_params.append((name, 'skip'))
        elif name == 'PCA__iterated_power':
            distributions[name] = IntUniformDistribution(1, 10)
            params[name] = val
        # PolynomialFeatures hyperparameters
        elif name == 'PolynomialFeatures__degree':
            continue
            # trial_params.append((name, trial.suggest_int(
            #     'PolynomialFeatures__degree', 2, 2)))
            # trial_params.append((name, 'skip'))
        elif name == 'PolynomialFeatures__include_bias':
            continue
            # trial_params.append((name, trial.suggest_categorical(
            #     'PolynomialFeatures__include_bias', ['False'])))
            # trial_params.append((name, 'skip'))
        elif name == 'PolynomialFeatures__interaction_only':
            continue
            # trial_params.append((name, trial.suggest_categorical(
            #     'PolynomialFeatures__interaction_only', ['False'])))
            # trial_params.append((name, 'skip'))
        # RBFSampler hyperparameters
        elif name == 'RBFSampler__gamma':
            distributions[name] = UniformDistribution(0.0, 1.0)
            params[name] = val
        # OneHotEncoder hyperparameters
        elif name == 'OneHotEncoder__minimum_fraction':
            distributions[name] = UniformDistribution(0.05, 0.25)
            params[name] = val
        elif name == 'OneHotEncoder__sparse':
            continue
            # trial_params.append((name, trial.suggest_categorical(
            #     'OneHotEncoder__sparse', ['False'])))
            # trial_params.append((name, 'skip'))
        elif name == 'OneHotEncoder__threshold':
            continue
            # trial_params.append((name, trial.suggest_int(
            #     'OneHotEncoder__threshold', 10, 10)))
            # trial_params.append((name, 'skip'))
        # ***** SELECTOR HYPERPARAMETERS *****
        # SelectFwe hyperparameters
        elif name == 'SelectFwe__alpha':
            distributions[name] = UniformDistribution(0, 0.05)
            params[name] = val
        # SelectPercentile hyperparameters
        elif name == 'SelectPercentile__percentile':
            distributions[name] = IntUniformDistribution(1, 99)
            params[name] = val
        # VarianceThreshold hyperparameters
        elif name == 'VarianceThreshold__threshold':
            distributions[name] = LogUniformDistribution(0.0001, 0.2)
            params[name] = val
        # SelectFwe hyperparameters
        elif name == 'SelectFromModel__threshold':
            # continue
            distributions[name] = UniformDistribution(0.0, 1.0)
            params[name] = val            
        elif name == 'SelectFromModel__ExtraTreesRegressor__n_estimators':
            continue
        elif name == 'SelectFromModel__ExtraTreesRegressor__max_features':
            distributions[name] = UniformDistribution(0, 1.0)
            params[name] = val
            # continue
        else:
            print("Unable to parse parameter name: " 
                  + name + ", skipping..")
            trial_params.append((name, 'skip'))
        
    trial = create_trial(params=params, 
                         distributions=distributions, 
                         value=value)
    
    return trial