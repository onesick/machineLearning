from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import time

class KNN():
    def __init__(self):
        self.model = KNeighborsClassifier()
        self.param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 10]}]

    def bestModel(self, **best_params):
        self.model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])

    def findBestK(self, X_train, y_train, param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]):
        self.param_grid = param_grid
        grid_search = GridSearchCV(self.model, param_grid, cv=5, verbose=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        return grid_search

    def fit(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()

        print('\n\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, X_test):
        return self.model.predict(X_test)    