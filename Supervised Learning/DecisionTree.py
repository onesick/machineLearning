import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# delete these two lines later
from sklearn import datasets, svm, metrics #TODO: later remove datasets
from sklearn.model_selection import train_test_split


class DecisionTree():
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 class_weight=None,
                 alpha=None,
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

    def fit(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()

        print('\n\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, X):
        return self.model.predict(X)        

    def set_params(self, **params):
        self.model.set_params(**params)
# digits = datasets.load_digits()

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

# plt.show()
# # flatten the images
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))

# # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001)

# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False
# )

# # Learn the digits on the train subset
# clf.fit(X_train, y_train)

# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test)

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

# plt.show()