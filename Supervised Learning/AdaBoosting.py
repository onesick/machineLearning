from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIR = 'images/boosting'

class AdaBoost():
    def __init__(self):
        self.model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=5), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)    
    
    
    def _plot_helper_(self, x_axis, train_scores, val_scores, train_label, val_label):
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

    def plot_learning_curve(self, learner, x_train, y_train, y_target, y_pred, **kwargs):
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

        
        _, train_scores, val_scores = learning_curve(learner.model, x_train, y_train,
                                                                     train_sizes=np.linspace(0.1, 1.0, 5),
                                                                     cv=kwargs['cv'], n_jobs=-1)

        

        # Create a new figure and plot learning curves both for the default and the optimal parameters
        plt.figure()
        self._plot_helper_(np.linspace(0.1, 1.0, 5), train_scores, val_scores,
                           train_label='Training', val_label='Cross-Validation')
       

        # Add title, legend, axes labels and eventually set y axis limits
        plt.title('{} - Learning Curves using {}-Fold Cross Validation'.format("adaBoosting", kwargs['cv']))
        plt.legend(loc='lower left')
        plt.xlabel('Training samples')
        plt.ylabel('Accuracy')
        if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)

        # Save figure
        plt.savefig(IMAGE_DIR + 'adaBoosting_learning_curve')

    def plot_decision_boundary(clf, X, y, axes=[0,9], alpha=0.5, contour=True):
        x1s = np.linspace(axes[0], axes[1], 1997)
        x2s = np.linspace(axes[0], axes[1], 1997)
        x1, x2 = np.meshgrid(x1s, x2s)
        X_new = np.c_[x1.ravel(), x2.ravel()]
        y_pred = clf.predict(X_new).reshape(x1.shape)
        custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
        plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
        if contour:
            custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
            plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
        plt.axis(axes)
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


    # following code is from hands on machine learning book from Oreily.
    # def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    #     x1s = np.linspace(axes[0], axes[1], 100)
    #     x2s = np.linspace(axes[2], axes[3], 100)
    #     x1, x2 = np.meshgrid(x1s, x2s)
    #     X_new = np.c_[x1.ravel(), x2.ravel()]
    #     y_pred = clf.predict(X_new).reshape(x1.shape)
    #     custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    #     plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    #     if contour:
    #         custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
    #         plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    #     plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    #     plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    #     plt.axis(axes)
    #     plt.xlabel(r"$x_1$", fontsize=18)
    #     plt.ylabel(r"$x_2$", fontsize=18, rotation=0)