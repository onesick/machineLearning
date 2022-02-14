from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, datasets
from sklearn.preprocessing import StandardScaler 

import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_sample_weight

IMAGE_DIR = 'images/'

class NeuralNetwork():
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 solver='lbfgs',
                 alpha=0.0001,
                 learning_rate_init=0.001,
                 max_iter=3
                ):
        # TODO: Try out the learner without preprocessed model VS raw model
        # self.model = 
        self.model = Pipeline([('normalized', StandardScaler()),
                                ('NeuralNetwork', MLPClassifier(
                                    hidden_layer_sizes=hidden_layer_sizes,
                                    activation=activation,
                                    solver=solver,
                                    alpha=alpha,
                                    learning_rate_init=learning_rate_init,
                                    max_iter=max_iter
                                    )
                                )]
                            )
    def fit(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print('fitting Neural Network\n')
        print('\n\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, X):
        return self.model.predict(X)    


    def cross_val(self, X_train, y_train, cv=5, scoring="accuracy"):
        cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)


    