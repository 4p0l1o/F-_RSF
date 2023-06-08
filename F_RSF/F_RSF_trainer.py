from plato.trainers import base

from functools import partial
import logging

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from plato.clients import simple
from plato.servers import fedavg
from plato.trainers import basic
import F_RSF_algorithm as CA

import pandas as pd
import numpy as np
#%matplotlib inline

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score
from plato.config import Config
import os
import pickle
import numpy
from sklearn.model_selection import RandomizedSearchCV
from plato.utils import csv_processor, fonts
import collections
from itertools import islice
import utils
from sksurv.linear_model.coxph import BreslowEstimator



class Trainer(base.Trainer):
    """A custom trainer with custom training and testing loops."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        callbacks: The callbacks that this trainer uses.
        """
        super().__init__()

        if model is None:
            self.model = partial(RandomSurvivalForest)
        else:
            self.model = model()

        self.train_loader = None
        self.sampler = None
        self._loss_criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.current_epoch = 0

        self.n_estimators = Config().trainer.n_estimators
        self.min_samples_split = Config().trainer.min_samples_split
        self.min_samples_leaf = Config().trainer.min_samples_leaf
        self.max_features = Config().trainer.max_features
        self.max_depth = Config().trainer.max_depth
        self.bootstrap = Config().trainer.bootstrap

        self.hyper_parameters = {}

    # pylint: disable=unused-argument
    def train_model(self, trainset, client_id, sampler, **kwargs):
        """A custom training loop."""

        logging.info("training model")
        if(not client_id in self.hyper_parameters):
            if(Config.trainer.do_crossValidation):
                if(client_id == "baseline"):
                    path = (
                        f"{Config().params['result_path']}/{Config.data.datasource}/params.csv"
                    )
                else:                
                    path = (
                        f"{Config().params['result_path']}/{Config.data.datasource}/params_{Config.clients.total_clients}_clients.csv"
                    )
                params = utils.get_params(path, client_id)
                if False :#params is not None:
                    self.hyper_parameters[client_id] = {
                    "n_estimators": int(params[1]),
                    "min_samples_split": int(params[2]),
                    "min_samples_leaf": int(params[3]),
                    "max_features": params[4],
                    "max_depth": int(params[5]),
                    "bootstrap": bool(params[6])
                    }
                    if self.hyper_parameters[client_id]["max_features"] == "":
                        self.hyper_parameters[client_id]["max_features"] = None
                else:
                    CV_params = self.crossValidation(trainset, client_id)
                    self.hyper_parameters[client_id] = CV_params
                    param_row = [
                        client_id,
                        CV_params["n_estimators"],
                        CV_params["min_samples_split"],
                        CV_params["min_samples_leaf"],
                        CV_params["max_features"],
                        CV_params["max_depth"],
                        CV_params["bootstrap"],
                    ]
                    #csv_processor.write_csv(path, param_row)
            else:
                self.hyper_parameters[client_id] = {
                "n_estimators": Config().trainer.n_estimators,
                "min_samples_split": Config().trainer.min_samples_split,
                "min_samples_leaf": Config().trainer.min_samples_leaf,
                "max_features": Config().trainer.max_features,
                "max_depth": Config().trainer.max_depth,
                "bootstrap": Config().trainer.bootstrap
                }

        rsf = RandomSurvivalForest(
            n_estimators=(self.hyper_parameters[client_id])["n_estimators"],
            min_samples_split=(self.hyper_parameters[client_id])["min_samples_split"],
            min_samples_leaf=(self.hyper_parameters[client_id])["min_samples_leaf"],
            max_features=(self.hyper_parameters[client_id])["max_features"],
            max_depth=(self.hyper_parameters[client_id])["max_depth"],
            n_jobs=-1,
            bootstrap=(self.hyper_parameters[client_id])["bootstrap"],
            random_state=20)
        
        
        rsf.fit(trainset[0], trainset[1])
        return rsf.estimators_, rsf.feature_names_in_, rsf.event_times_, rsf.n_outputs_
    
    def pickBestTrees(self, weights, trainset, testset, number_of_trees):
        rsf = RandomSurvivalForest()
        rsf.feature_names_in_ = weights[1]
        rsf.event_times_ = weights[2]
        rsf.n_outputs_ = weights[3]
        ordered_trees_ci = collections.OrderedDict()
        ordered_trees_ibs = collections.OrderedDict()

        ordered_trees_ibs = self.order_by_ibs(rsf, weights, trainset, testset)
        ordered_trees_ci = self.order_by_ci(rsf, weights, trainset, testset)
        bestTrees_ibs = self.take(number_of_trees, ordered_trees_ibs.values())
        bestTrees_ci = self.take(number_of_trees, ordered_trees_ci.values())

        return bestTrees_ibs, bestTrees_ci, rsf.feature_names_in_, rsf.event_times_, rsf.n_outputs_

    def order_by_ibs(self, rsf, weights, trainset, testset):
        test_time = []
        for time in testset[1]:
            test_time.append(time[1])
        #lower, upper = np.percentile(times, [10, 90])
        lower_event_time = min(rsf.event_times_)
        lower_test_time = min(test_time)
        lower = max([lower_event_time, lower_test_time])
        upper_event_time = max(rsf.event_times_)
        upper_test_time = max(test_time)
        lower = max([lower_event_time, lower_test_time])
        upper = min([upper_event_time, upper_test_time])
        times = np.arange(lower+1, upper-1 + 1)
        #times = rsf.event_times_
        trees = {}
        for tree in weights[0]:
            rsf.estimators_ = [tree]
            #try:
            rsf_surv_prob = np.row_stack([
            fn(times)
                for fn in rsf.predict_survival_function(testset[0])
            ])
            y = np.concatenate((trainset[1], testset[1]))
            ibs = integrated_brier_score(y, testset[1], rsf_surv_prob, times)
            trees[ibs] = tree
            
        ordered_trees = collections.OrderedDict(sorted(trees.items()))
        return ordered_trees
    
    def order_by_ci(self, rsf, weights, trainset, testset):
        trees = {}
        for tree in weights[0]:
            rsf.estimators_ = [tree]

            ci = rsf.score(testset[0], testset[1])
            trees[ci] = tree
        ordered_trees = collections.OrderedDict(sorted(trees.items(), reverse=True))
        return ordered_trees

    def take(self, n, iterable):
        "Return first n items of the iterable as a list"
        return list(islice(iterable, n))

    def load_model(self, filename=None, location=None):
        return
    
    def save_model(self, filename=None, location=None):
        return
    
    def test(self, testset, trainset, weights, sampler=None, **kwargs) -> float:

        rsf = RandomSurvivalForest()
        #rsf.predict_survival_function()
        #rsf.
        rsf.estimators_ = weights[0]
        # for tree in rsf.estimators_:
        #     tree.n_outputs_=6
        #print((rsf.estimators_[0]).n_outputs_)
        rsf.feature_names_in_ = weights[1]
        rsf.event_times_ = weights[2]
        #print(weights[3])
        rsf.n_outputs_ = weights[3]
        
        print("We are testing!")

        print("model score is: ")
        score = rsf.score(testset[0], testset[1])
        print(score)
        ibs = self.calculate_Brier_Score(rsf, testset, trainset)
        #print(ibs)
        return score, ibs
    
    def calculate_Brier_Score(self, rsf, testset, trainset):
            
            test_time = []
            for time in testset[1]:
                test_time.append(time[1])
            lower_event_time = min(rsf.event_times_)
            lower_test_time = min(test_time)
            lower = max([lower_event_time, lower_test_time])
            upper_event_time = max(rsf.event_times_)
            upper_test_time = max(test_time)
            lower = max([lower_event_time, lower_test_time])
            upper = min([upper_event_time, upper_test_time])
            times = np.arange(lower+1, upper-1 + 1)

            t_train = testset[1]['time']
            e_train = testset[1]['event']
            RandomSurvivalForest(rsf)
            test_predictions = rsf.predict(trainset[0])
            test_pred = []
            for pred in test_predictions:
                # if pred == 0:
                #     pred = 0.000000000000001
                test_pred.append(1/pred)
            breslow = BreslowEstimator().fit(test_pred, e_train, t_train)
            test_predictions = rsf.predict(testset[0])
            test_pred = []
            for pred in test_predictions:
                # if pred == 0:
                #     pred = 0.000000000000001
                test_pred.append(1/pred)
            test_surv_fn = breslow.get_survival_function(test_pred)
        
            surv_preds = np.row_stack([fn(times) for fn in test_surv_fn])
            surv_preds = numpy.nan_to_num(surv_preds, copy=True, nan=1.0, posinf=None, neginf=None)
            y = np.concatenate((trainset[1], testset[1]))
            ibs = integrated_brier_score(y, testset[1], surv_preds, times)
            print(ibs)

            return ibs
    
    def testClient(self, testset, trainset, weights, sampler=None, **kwargs) -> float:
        rsf = RandomSurvivalForest()
        rsf.estimators_ = (weights[0])
        rsf.feature_names_in_ = (weights[1])
        rsf.event_times_ = weights[2]
        rsf.n_outputs_ = weights[3]
        
        print("We are testing the client!")

        print("model score is: ")
        score = rsf.score(testset[0], testset[1])
        print(score)
        ibs = self.calculate_Brier_Score(rsf, testset, trainset)
        return score, ibs
    
    def test_all_clients(self, datasets, weights):
        print("testing all clients!")
        rsf = RandomSurvivalForest()
        rsf.estimators_ = (weights[0])
        rsf.feature_names_in_ = (weights[1])
        rsf.event_times_ = weights[2]
        rsf.n_outputs_ = weights[3]
        for id in range(1, Config.clients.total_clients+1):
            X_train, y_train, X_test, y_test = datasets[id]
            score = rsf.score(X_test, y_test)
            print(score)
            ibs = self.calculate_Brier_Score(rsf, [X_test, y_test], [X_train, y_train])
            print(ibs)
            path = (
                f"{Config().params['result_path']}/{Config.data.datasource}/accuracy_{Config.clients.total_clients}_clients_{int(Config.server.n_trees/Config.clients.total_clients)}_trees.csv"
            )
            accuracy_row = [
                weights[4] + "_client_" + str(id),
                score,
                ibs
            ]
            csv_processor.write_csv(path, accuracy_row)
    
    def train(self, trainset, client_id, sampler, **kwargs) -> float:
        training_time = 0
        weights = self.train_model(trainset, client_id, sampler)
        return weights
        #return super().train(trainset, sampler, **kwargs)
    
    def crossValidation(self, trainset, client_id):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
        # Number of features to consider at every split
        max_features = ["sqrt", "log2"]
        # Maximum number of levels in tree
        max_depth = [2, 5, 10, 20, 40]
        #max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 4, 6, 8, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 3, 4, 5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        

        rsf = RandomSurvivalForest()

        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rsf_random = RandomizedSearchCV(estimator = rsf, param_distributions = random_grid, verbose=2, n_iter = Config.trainer.iter_rounds, cv = 3, n_jobs = -1)
        # Fit the random search model
        rsf_random.fit(trainset[0], trainset[1])
        logging.info(
            fonts.colourize(
                f"[{client_id}] Hyper paramters: {rsf_random.best_params_}%\n"
            )
        )

        return rsf_random.best_params_