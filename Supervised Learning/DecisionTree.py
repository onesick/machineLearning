import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DecisionTree():
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0,
                 class_weight=None,
                 alpha=0.0001,
                 verbose=False):
        print("initiating Decision Tree")
        self.model = DecisionTreeClassifier(
                 criterion=criterion,
                 splitter=splitter,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 random_state=random_state,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 class_weight=class_weight,
                )
        self._alpha=alpha
    def fit(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        self.prune(X_train, y_train)
        print('\n\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, X):
        return self.model.predict(X)    

    def remove_subtree(self, root):
        """
        Clean up
        :param root:
        :return:
        """
        tmp_tree = self.model.tree_
        visited, stack = set(), [root]
        while stack:
            v = stack.pop()
            visited.add(v)
            left = tmp_tree.children_left[v]
            right = tmp_tree.children_right[v]
            if left >= 0:
                stack.append(left)
            if right >= 0:
                stack.append(right)
        for node in visited:
            tmp_tree.children_left[node] = -1
            tmp_tree.children_right[node] = -1
        return
    
    # pruning code from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
    def prune(self, X_train, y_train):
        c = 1 - self._alpha
        if self._alpha <= -1:  # Early exit
            return self
        tmp_tree = self.model.tree_
        best_score = self.model.score(X_train, y_train)
        candidates = np.flatnonzero(tmp_tree.children_left >= 0)
        for candidate in reversed(candidates):  # Go backwards/leaves up
            if tmp_tree.children_left[candidate] == tmp_tree.children_right[candidate]:  # leaf node. Ignore
                continue
            left = tmp_tree.children_left[candidate]
            right = tmp_tree.children_right[candidate]
            tmp_tree.children_left[candidate] = tmp_tree.children_right[candidate] = -1
            score = self.model.score(X_train, y_train)
            if score >= c * best_score:
                best_score = score
                self.remove_subtree(candidate)
            else:
                tmp_tree.children_left[candidate] = left
                tmp_tree.children_right[candidate] = right
        assert (self.model.tree_.children_left >= 0).sum() == (self.model.tree_.children_right >= 0).sum()

        return self
