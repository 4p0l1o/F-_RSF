"""
This example uses a very simple model and the MNIST dataset to show how the model,
the training and validation datasets, as well as the training and testing loops can
be customized in Plato.
"""
from functools import partial

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from plato.clients import simple
from plato.servers import fedavg
import F_RSF_algorithm as FSF_CA
import F_RSF_trainer as FSF_T
import F_RSF_dataSource as FSF_DS
import F_RSF_server as FSF_S
import F_RSF_client as FSF_C

import pandas as pd
import numpy as np
#%matplotlib inline

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest


def main():
    """
    A Plato federated learning training session using a custom model,
    datasource, and trainer.
    """
    model = partial(RandomSurvivalForest)
    datasource = FSF_DS.DataSource
    trainer = FSF_T.Trainer
    algorithm = FSF_CA.Algorithm

    client = FSF_C.Client(model=model, trainer=trainer, algorithm=algorithm, datasource=datasource)
    server = FSF_S.Server(model=model, trainer=trainer, algorithm=algorithm, datasource=datasource)
    server.run(client)


if __name__ == "__main__":
    main()
