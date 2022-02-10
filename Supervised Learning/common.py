from sklearn.model_selection import train_test_split, learning_curve, validation_curve
import matplotlib.pyplot as plt
from sklearn import metrics, datasets
from sklearn.preprocessing import StandardScaler 
from DecisionTree import DecisionTree
from NeuralNetwork import NeuralNetwork

import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_sample_weight

IMAGE_DIR = 'images/'
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)


def f1_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return f1_score(truth, pred, average="binary", sample_weight=wts)


# _plot_helper and plot_learning_curve(with my modification) are from
# https://github.com/ezerilli/Machine_Learning/blob/master/Supervised_Learning/

def _plot_helper_(x_axis, train_scores, val_scores, train_label, val_label):
    """Plot helper.

        Args:
            x_axis (ndarray): x axis array.
            train_scores (ndarray): array of training scores.
            val_scores (ndarray): array of validation scores.
            train_label (string): training plot label.
            val_label (string): validation plot label.

        Returns:
            None.
        """

    # Compute training and validation scores mean and standard deviation over cross-validation folds.
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plot training and validation mean by filling in between mean + std and mean - std
    train_plot = plt.plot(x_axis, train_scores_mean, '-o', markersize=2, label=train_label)
    val_plot = plt.plot(x_axis, val_scores_mean, '-o', markersize=2, label=val_label)
    plt.fill_between(x_axis, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                        alpha=0.1, color=train_plot[0].get_color())
    plt.fill_between(x_axis, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std,
                        alpha=0.1, color=val_plot[0].get_color())

def plot_learning_curve(learner, x_train, y_train, **kwargs):
        """Plot learning curves with cross-validation.
            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for learning curves plotting:
                    - train_sizes (ndarray): training set sizes to plot the learning curves over.
                    - cv (int): number of k-folds in cross-validation.
                    - y_lim (float): lower y axis limit.
            Returns:
               None.
            """
        print('\n\nLearning Analysis with k-Fold Cross Validation')
        # Clone the model and use clone for training in learning curves
        

        # Set default parameters and produce corresponding learning curves using k-fold cross-validation
        if 'scorer' in kwargs:
            scorer = kwargs['scorer']
        else:
            scorer = make_scorer(balanced_accuracy)
        _, train_scores_default, val_scores_default = learning_curve(learner.model, x_train, y_train,
                                                                     train_sizes=kwargs['train_sizes'],
                                                                     cv=kwargs['cv'], n_jobs=-1, scoring=scorer)

        

        # Create a new figure and plot learning curves both for the default and the optimal parameters
        plt.figure()
        _plot_helper_(kwargs['train_sizes'], train_scores_default, val_scores_default,
                           train_label='Training with default params', val_label='Cross-Validation, default params')
       

        # Add title, legend, axes labels and eventually set y axis limits
        plt.title('{} - Learning Curves using {}-Fold Cross Validation'.format(learner.model, kwargs['cv']))
        plt.legend(loc='lower left')
        plt.xlabel('Training samples')
        plt.ylabel('Accuracy')
        if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)

        # Save figure
        plt.savefig(IMAGE_DIR + '{}_learning_curve'.format(learner.model))

def plot_model_complexity(learner, x_train, y_train, param_name, param_range, xlabel='Max Tree Depth', ylabel='Accuracy', xscale='linear', **kwargs):
    train_scores, val_scores = validation_curve(learner.model, x_train, y_train,
                                                 param_name=param_name, param_range=param_range, cv=kwargs['cv'])
    plt.figure()
    _plot_helper_(param_range, train_scores, val_scores, "Training", "Cross Validation")
    plt.title('{} - Validation curves using {}-Fold Cross Validation'.format(learner.model, kwargs['cv']))
    plt.legend(loc='lower left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(xscale)
    if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)
    # Save figure
    plt.savefig(IMAGE_DIR + '{}_model_complexity'.format(learner.model))


def performExperiments(data, clf, train_size=np.linspace(0.1, 1.0, 5), cv=5, y_lim=0.4, param_name="max_depth", param_range=list(range(1, 50))):
    print("Starting experiments on {}".format(clf))
    n_samples = len(data.images)

    X_train, X_test, y_train, y_test = train_test_split(
    data.images.reshape((n_samples, -1)), data.target, test_size=0.2, shuffle=True
    )

    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    # plot learning curve

    f1_scorer = make_scorer(f1_accuracy) # later, if I want to pass f1 scorer, I send this out
    plot_learning_curve(clf, X_train, y_train, cv=cv, y_lim=y_lim, train_sizes=train_size)

    plot_model_complexity(clf, X_train, y_train, param_name, param_range, cv=cv, y_lim=y_lim)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    print(
        f"Confusion matrix report for classifier {clf}:\n"
        f"{confusion_matrix(y_test, predicted)}"
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='start of experiment')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    args = parser.parse_args()
    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1, dtype='uint64')
        print("Using seed {}".format(seed))

    # Instantiate decision tree
    dtLearner = DecisionTree()
    # Instantiate Neural Net
    nerualNet = NeuralNetwork()

    # loading data
    data = datasets.load_digits()
    # print(pd.DataFrame(data.data).head())
    # print(plt.imshow(data.images[0]))
    performExperiments(data, dtLearner)
    # performExperiments(data, nerualNet)

    # TODO: implement neural network parameters for analysis
    
