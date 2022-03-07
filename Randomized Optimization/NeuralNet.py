import mlrose_hiive
import numpy as np
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import matplotlib.pyplot as plt

# loading data
data = datasets.load_digits()
n_samples = len(data.images)
X_train, X_test, y_train, y_test = train_test_split(
    data.images.reshape((n_samples, -1)), data.target, test_size=0.2, shuffle=True
    )


rhc_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.01, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 3)

sa_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.01, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 3)

ga_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.01, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 3)

gradient_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.01, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()


#rhc model
rhc_model.fit(X_train_scaled, y_train_hot)

y_train_pred = rhc_model.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print(y_train_accuracy)
y_test_pred = rhc_model.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print(y_test_accuracy)
#sa model
sa_model.fit(X_train_scaled, y_train_hot)

y_train_pred = sa_model.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print(y_train_accuracy)
y_test_pred = sa_model.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print(y_test_accuracy)
# ga model
ga_model.fit(X_train_scaled, y_train_hot)

y_train_pred = ga_model.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print(y_train_accuracy)
y_test_pred = ga_model.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print(y_test_accuracy)
#gradient model
gradient_model.fit(X_train_scaled, y_train_hot)

y_train_pred = gradient_model.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print(y_train_accuracy)
y_test_pred = gradient_model.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print(y_test_accuracy)
