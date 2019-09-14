import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from time import time


def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02, limits=False, alpha=0.05,
                          minx=-0.5, maxx=1, miny=-0.8, maxy=2):
    # setup marker generator and color map
    markers = ('s', 'o', 'x', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=alpha, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')

    if limits:
        ax = plt.gca()
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)


def fit_class(X, y, test_size=0.3, stratify_y=True, scale=None,
              classifier='lr', xlabel="x1", ylabel="y2",
              lr_c=100.0, perc_max_iter=40, perc_eta=0.1,
              svm_l_c=1.0, svm_k_c=1.0, svm_k_gamma=0.2,
              tree_criterion='gini', tree_max_depth=4,
              forest_criterion='gini', forest_n_estimators=25, forest_njobs=2,
              knn_nn=5, knn_p=2, knn_metric='minkowski',
              random_state=1, plot_result='show', plot_resolution=0.02, save_path=""):
    if stratify_y:
        stratify = y
    else:
        stratify = None
    print("\n----- Fitting classification algorithms to predict", y.name, "from", xlabel, ylabel,
          "\n\nTotal samples in the dataset: {0:,}".format(len(X)))
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=stratify)
    print('Labels counts in y_train:', np.bincount(y_train),
          '\nLabels counts in y_test:', np.bincount(y_test),
          '\nLabels counts in y:', np.bincount(y))

    if scale:
        if scale == 'norm':
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif scale == 'std':
            scaler = StandardScaler()
        else:
            raise AttributeError("Parameter 'scale' must be set to either 'norm' or 'std'.")
        scaler.fit(X_train)
        X = pd.DataFrame(scaler.transform(X))
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        print("\n --- Features scaled using StandardScaler.")

    models = {
        'lr': LogisticRegression(C=lr_c, random_state=random_state),
        'perc': Perceptron(max_iter=perc_max_iter, eta0=perc_eta, random_state=random_state),
        'svm_linear': SVC(kernel='linear', C=svm_l_c, random_state=random_state),
        'svm_kernel': SVC(kernel='rbf', C=svm_k_c, gamma=svm_k_gamma, random_state=random_state),
        'tree': DecisionTreeClassifier(criterion=tree_criterion,
                                       max_depth=tree_max_depth,
                                       random_state=random_state),
        'forest': RandomForestClassifier(criterion=forest_criterion,
                                         n_estimators=forest_n_estimators,
                                         n_jobs=forest_njobs,
                                         random_state=random_state),
        'knn': KNeighborsClassifier(n_neighbors=knn_nn, p=knn_p,
                                    metric=knn_metric)
    }
    if classifier == 'all':
        for name, class_model in models.items():
            model = class_model
            print("\n----- Fitting", name.upper())
            t = time()
            model.fit(X, y)
            plot_decision_regions(X, y, classifier=model, alpha=0.3, result=plot_result, name=name,
                                  xlabel=xlabel, ylabel=ylabel, save_path=save_path,
                                  resolution=plot_resolution,
                                  title=name.upper() + " classification algorithm, "
                                                       "\ndecision boundary"
                                                       "\nx1: {0}, x2: {1}".format(xlabel, ylabel))
            elapsed = time() - t
            print("\n - took {0:.2f} seconds.".format(elapsed))
    elif type(classifier) == list:
        for classi in classifier:
            try:
                model = models[classi]
            except (KeyError, TypeError):
                raise AttributeError("Parameter 'classifier' must be string or list of strings with one or several of "
                                     "'lr', 'perc', 'svm_linear', 'svm_kernel', 'tree', 'forest', 'knn', or 'all'")
            print("\n----- Fitting", classi.upper())
            t = time()
            model.fit(X, y)
            plot_decision_regions(X, y, classifier=model, alpha=0.3, result=plot_result, name=classi,
                                  xlabel=xlabel, ylabel=ylabel, save_path=save_path,
                                  resolution=plot_resolution,
                                  title=classi.upper() + " classification algorithm, "
                                                         "\ndecision boundary"
                                                         "\nx1: {0}, x2: {1}".format(xlabel, ylabel))
            elapsed = time() - t
            print("\n - took {0:.2f} seconds.".format(elapsed))
    else:
        try:
            model = models[classifier]
        except (KeyError, TypeError):
            raise AttributeError("Parameter 'classifier' must be string or list of strings with one or several of "
                                 "'lr', 'perc', 'svm_linear', 'svm_kernel', 'tree', 'forest', 'knn', or 'all'")
        print("\n----- Fitting", classifier.upper())
        t = time()
        model.fit(X, y)
        plot_decision_regions(X, y, classifier=model, alpha=0.3, result=plot_result, name=classifier,
                              xlabel=xlabel, ylabel=ylabel, save_path=save_path,
                              resolution=plot_resolution,
                              title=classifier.upper() + " classification algorithm, "
                                                         "\ndecision boundary"
                                                         "\nx1: {0}, x2: {1}".format(xlabel, ylabel))
        elapsed = time() - t
        print("\n - took {0:.2f} seconds.".format(elapsed))

#
# dot_data = export_graphviz(model,
#                           filled=True,
#                           rounded=True,
#                           class_names=['def',
#                                        'for'],
#                           feature_names=[xcol1,
#                                          xcol2],
#                           out_file=None)
# graph = graph_from_dot_data(dot_data)
# graph.write_png('img/tree.png')
