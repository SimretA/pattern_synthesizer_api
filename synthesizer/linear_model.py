import pandas as pd
import torch

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.metrics import precision_recall_fscore_support

class LinearPatternsModel:
    def __init__(self, theme, data):
        self.theme = theme
        self.patterns = []
        self.weights = []
        self.nnModel = None
        self.data = data
    

    def reconfigure(self):
        pass


    def train_linear_model(self):
        pass
