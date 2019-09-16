import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import norm, shapiro, normaltest
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


def fit_model(model, model_name, X_train, y_train, X_test, y_test, X_val1, y_val1, X_val2, y_val2,
              return_coefs=False):
    t = time()

    # fit the model
    model.fit(X_train, y_train)

    # make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_val1 = model.predict(X_val1)
    y_pred_val2 = model.predict(X_val2)

    # score model performance
    train_score = accuracy_score(y_train, y_pred_train)
    test_score = accuracy_score(y_test, y_pred_test)
    val1_score = accuracy_score(y_val1, y_pred_val1)
    val2_score = accuracy_score(y_val2, y_pred_val2)

    elapsed = time() - t
    print("\n{0} fit, took {1:,.2f} seconds ({2:,.2f} minutes)".format(model_name, elapsed, elapsed / 60) +
          "\naccuracy: train={0:.2f}, test={1:.2f}, validation #1={2:.2f}, validation #2={3:.2f}"
          .format(train_score, test_score, val1_score, val2_score))

    if return_coefs:
        return model.coef_[0]


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


def fit_norm_dist(series, h_bins='auto',
                  create_figure=True, ax=None,
                  show_plot=True, suptitle=None,
                  title=None, xlabel=None, ylabel='Distribution',
                  figsize=(6, 6), lab2='from_mean',
                  mean_lift=0.99, std_lift=1.007,
                  sig_lift=0.03, per_lift=0.1, val_lift=0.23,
                  x_between=None, x_min=None, x_max=None,
                  t_shapiro=True, t_k2=True, t_anderson=True, alpha=0.05):
    """
    :param series: pandas Series
        Series to be plotted
    :param h_bins: int
        number of bins to for histogram (default='auto')
    :param create_figure: Boolean
        whether to initialize matplotlib figure and axis
        (default=True)
    :param ax: matplotlib axis
        matplotlib axis to plot on (for multi-plot figures)
    :param show_plot: Boolean
        whether to show plot at the end (default=True)
    :param suptitle: string
        string to use for plot suptitle
    :param title: string
        string to use for plot title
    :param xlabel: string
        string to use for x axis label
    :param ylabel: string
        string to use for y axis label
    :param figsize: tuple(float, float)
        size of the figure
    :param lab2: string (must be 'cdf' or 'from_mean'
        which percentage values to display (CDF or from mean)
        (default='from_mean')
    :param mean_lift: float
        lift for mean caption
    :param std_lift: float
        lift for std caption
    :param sig_lift: float
        lift for sigma captions
    :param per_lift: float
        lift for percentage captions
    :param val_lift: float
        lift for values captions
    :param x_min: float
        estimate probability that x > x_min
    :param x_max: float
        estimate probability that x < x_max
    :param x_between: tuple(float, float)
        estimate probability that x_min < x < x_max
    :return: series.describe()
        statistical summary of the plotted Series
    """
    if create_figure:
        f, ax = plt.subplots(1, figsize=figsize)
        if suptitle:
            f.suptitle(suptitle)
    # plot data
    n, bins, patches = plt.hist(x=series, bins=h_bins)
    mu = series.mean()
    sigma = series.std()
    # initialize a normal distribution
    nd = norm(mu, sigma)
    # plot mean std
    ax.axvline(mu, color='black', linestyle='--')
    ax.text(mu * mean_lift, n.max() * 0.4,
            "Mean: {0:.2f}".format(mu),
            rotation='vertical')
    ax.text(mu * std_lift, n.max() * 0.4,
            "StDev: {0:.2f}".format(sigma),
            rotation='vertical')
    # generate sigma lines and labels
    i = np.arange(-3, 4)
    vlines = mu + i * sigma
    labels1 = pd.Series(i).astype('str') + '$\sigma$'
    if lab2 == 'cdf':
        labels2 = pd.Series(nd.cdf(vlines) * 100) \
                      .round(2).astype('str') + '%'
    elif lab2 == 'from_mean':
        labels2 = pd.Series(abs(50 - nd.cdf(vlines) * 100) * 2) \
                      .round(2).astype('str') + '%'
    else:
        raise AttributeError("Parameter 'lab2' must be either set to 'cdf' or 'from_mean'")
    labels2 = labels2.astype('str')

    # plot sigma lines and labels
    for vline, label1, label2 in zip(vlines, labels1, labels2):
        # plot sigma lines
        if vline != mu:
            ax.axvline(vline, linestyle=':', color='salmon')
            ax.text(vline, n.max() * sig_lift, label1)
        ax.text(vline, n.max() * per_lift, label2, rotation=45)
        ax.text(vline, n.max() * val_lift,
                round(vline, 2), rotation=45)

    # fit a normal curve
    # generate x in range of mu +/- 5 sigma
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma,
                    1000)
    # calculate PDF
    y = nd.pdf(x)
    # plot fitted distribution
    ax2 = ax.twinx()
    ax2.plot(x, y, color='red', label='Fitted normal curve')
    ax2.legend(loc='best')
    ax2.set_ylim(0)

    if not xlabel:
        xlabel = series.name
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if not title:
        title = "Histogram of {0}".format(series.name)
    ax.set_title(title)

    ax.grid(False)
    ax2.grid(False)

    if x_min:
        # estimate probability that x > x_min
        p = (1 - nd.cdf(x_min)) * 100
        ax.text(mu - 5 * sigma, n.max() * 0.9, "$P ~ ( X > {0} ) ~ = $".format(x_min) +
                "\n$ = {0:.2f}\%$".format(p))
        ax2.fill_between(x, 0, y, color='red', alpha=0.2,
                         where=x > x_min)
        mask = series > x_min
        print("Using normal distribution, from the total {0:,} records in the Series, "
              "{1:,.0f} are expected to have {2} > {3}"
              .format(len(series), len(series) * p / 100, series.name, x_min),
              "\n\nActual number of records with {0} > {1}: {2:,}"
              .format(series.name, x_min, len(series[mask])))
    elif x_max:
        # estimate probability that x > x_min
        p = nd.cdf(x_max) * 100
        ax.text(mu - 5 * sigma, n.max() * 0.9, "$P ~ ( X < {0} ) ~ = $".format(x_max) +
                "\n$ = {0:.2f}\%$".format(p))
        ax2.fill_between(x, 0, y, color='red', alpha=0.2,
                         where=x < x_max)
        mask = series < x_max
        print("Using normal distribution, from the total {0:,} records in the Series, "
              "{1:,.0f} are expected to have {2} < {3}"
              .format(len(series), len(series) * p / 100, series.name, x_max),
              "\n\nActual number of records with {0} < {1}: {2:,}"
              .format(series.name, x_max, len(series[mask])))
    elif x_between:
        # estimate probability that x_min < x < x_max
        x_min, x_max = x_between
        p = (nd.cdf(x_max) - nd.cdf(x_min)) * 100
        ax.text(mu - 5 * sigma, n.max() * 0.9, "$P ~ ( {0} < X < {1} ) ~ = $".format(x_min, x_max) +
                "\n$ = {0:.2f}\%$".format(p))
        ax2.fill_between(x, 0, y, color='red', alpha=0.2,
                         where=np.logical_and(x > x_min, x < x_max))
        mask = np.logical_and(series < x_max, series > x_min)
        print("Using normal distribution, from the total {0:,} records in the Series, "
              "{1:,.0f} are expected to have {2} < {3} < {4}"
              .format(len(series), len(series) * p / 100, x_min, series.name, x_max),
              "\n\nActual number of records with {0} < {1} < {2}: {3:,}"
              .format(x_min, series.name, x_max, len(series[mask])))

    if show_plot:
        plt.show()

    if t_shapiro:
        stat, p = shapiro(series.dropna())
        print("\n----- Shapiro-Wilks normality test results:\nW = {0:.3f}, p-value = {1:.3f}"
              .format(stat, p))
        # interpret
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')

    if t_k2:
        stat, p = normaltest(series.dropna())
        print("\n----- D’Agostino’s K^2 normality test results:"
              "\ns^2 + k^2 = {0:.3f}, p-value = {1:.3f},"
              .format(stat, p) + "\nwhere s is the z-score returned "
                                 "by skewtest and k is the z-score returned by kurtosistest"
                                 "\nand p-value is a 2-sided chi squared probability for the hypothesis test.")
        # interpret
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')

    if t_anderson:
        result = anderson(series.dropna())
        print("\n----- Anderson-Darling Test results:"
              "\nStatistic: {0:.3f}".format(result.statistic))
        p = 0
        print("Sigificance level: critical value")
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('{0:.3f}: {1:.3f}, data looks normal (fail to reject H0)'.format(sl, cv))
            else:
                print('{0:.3f}: {1:.3f}, data does not look normal (reject H0)'.format(sl, cv))

    return series.describe()


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
