from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

class NeuralNetwork():
    def __init__(self,
                 hidden_layer_sizes=(10,),
                 activation='relu',
                 solver='lbfgs'
                ):
        # TODO: Try out the learner without preprocessed model VS raw model
        # self.model = 
        self.model = Pipeline([('normalized', StandardScaler()),
                                ('NeuralNetwork', MLPClassifier(
                                    hidden_layer_sizes=hidden_layer_sizes,
                                    activation=activation,
                                    solver=solver
                                    )
                                )]
                            )

    def fit(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print('fitting Neural Network\n')
        print('\n\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))