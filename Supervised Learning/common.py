from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics, datasets
from sklearn.preprocessing import StandardScaler 
from DecisionTree import DecisionTree

import argparse
import numpy as np

from sklearn.pipeline import Pipeline



def performDTExperiments(data, clf):
    print("Starting experiments on {}".format(clf))
    n_samples = len(data.images)

    X_train, X_test, y_train, y_test = train_test_split(
    data.images.reshape((n_samples, -1)), data.target, test_size=0.2, shuffle=True
    )

    pipe = Pipeline([('Scale', StandardScaler()), ('DecisionTree', clf)])
    # TODO: first, search with no param optimization, but start later
    pipe.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='start of experiment')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    args = parser.parse_args()
    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1, dtype='uint64')
        print("Using seed {}".format(seed))

    dtLearner = DecisionTree()

    # loading data
    data = datasets.load_digits()

    performDTExperiments(data, dtLearner)