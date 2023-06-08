from plato.datasources import base

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
import logging

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from plato.config import Config
from survdata import datasets
from plato.utils import csv_processor, fonts
import random


class DataSource(base.DataSource):
    """
    A custom datasource with custom training and validation datasets.
    """

    def __init__(self):
        super().__init__()

        X, y = self.getDataSet()

        transform = OneHotEncoder()
        Xt = transform.fit_transform(X)
        Xtt = Xt.fillna(Xt.mean())

        seed = random.randint(0,10000)
        Xtt, X_test, y, y_test = train_test_split(
            Xtt, y, test_size=Config.data.test_size)
        self.testset = [X_test, y_test]
        self.trainset = [Xtt, y]
        self.trainsets = []
        self.datasets = {}
        for i in range(Config.clients.total_clients, 1, -1):
            X_train, X_test, y_train, y_test = train_test_split(
                Xtt, y, test_size=1-1/i, random_state=0)
            train = [X_train, y_train]

            self.trainsets.extend(train)
            if(i==2):
                train = [X_test, y_test]
                self.trainsets.extend(train)
            else:
                Xtt = X_test
                y = y_test
        for id in range(1, Config.clients.total_clients+1):
            self.get_client_dataset(id)

    def targets(self, client_id):
        """ Obtains a list of targets (labels) for all the examples
        in the dataset. """
        return self.get_client_dataset(client_id)[1]

    def getDataSet(self):
        print(Config().data.datasource)
        if Config().data.datasource == "AIDS":
            return datasets.load_aids_dataset()
        elif Config().data.datasource == "gbsg2":
            return datasets.load_gbsg2_dataset()
        elif Config().data.datasource == "flchain":
            return datasets.load_flchain_dataset()
        elif Config().data.datasource == "metabric":
            return datasets.load_metabric_dataset()
        elif Config().data.datasource == "nhanes":
            return datasets.load_nhanes_dataset()
        elif Config().data.datasource == "seer":
            return datasets.load_seer_dataset()
        elif Config().data.datasource == "support":
            return datasets.load_support_dataset()
        elif Config().data.datasource == "veterans":
            return datasets.load_veterans_dataset()
        elif Config().data.datasource == "whas500":
            print("Here")
            return datasets.load_whas500_dataset()


    def get_dataset(self, client_id):
        return self.get_client_dataset(client_id)
    
    def get_all_clients_dataset(self):
        return self.datasets
        
    def get_client_dataset(self, client_id):
        if client_id in self.datasets:
            return (self.datasets[client_id])
        else:
            X = self.trainsets[(client_id*2)-2]
            y = self.trainsets[(client_id*2)-1]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.data.test_size, random_state=0)
            dataset = [X_train, y_train, X_test, y_test]
            self.datasets[client_id] = dataset
            return self.datasets[client_id]
        
    def num_train_examples(self, client_id) -> int:
        """ Obtains the number of training examples. """
        return len(self.get_client_dataset(client_id)[0])
    
    def get_test_set(self):
        return self.testset
    def get_train_set(self):
        return self.trainset