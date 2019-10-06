import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from itertools import combinations
from sklearn import clone
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy.stats import norm, shapiro, normaltest, anderson
from time import time


class SBS():
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.3, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test,
                    indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


def get_fit_times(clf, feat_dict, target_dict, model_name, model_code, n_jobs):
    tt = time()

    times_scores = dict()

    for feats in feat_dict.keys():
        times_scores[feats] = dict()
        t = time()

        model = clone(clf)
        model.fit(feat_dict[feats]['train'], target_dict['train'])

        times_scores[feats]['acc'] = model.score(feat_dict[feats]['test'], target_dict['test'])
        times_scores[feats]['fit_time'] = time() - t
        times_scores[feats]['n_jobs'] = n_jobs

    model_times_scores_df = pd.DataFrame(times_scores).reset_index().rename(columns={'index': 'result'})
    idx = pd.Index([model_code for i in range(len(model_times_scores_df))])
    model_times_scores_df.index = idx

    elapsed = time() - tt

    print("{0} fit, took {1:,.2f} seconds ({2:,.2f} minutes) in total".format(model_name, elapsed, elapsed / 60))

    return model_times_scores_df


def fit_sbs(classifier, k_features, X, y, y_min=None, y_max=None, height=4, width=4,
            title="SBS", output='show', save_path='sbs.png', return_feats=True):
    t = time()

    sbs = SBS(classifier, k_features=k_features)

    sbs.fit(X, y)

    elapsed = time() - t
    print("Sequential Backwards Selection algorithm was applied. Took {0:,.2f} seconds, ({1:,.2f} minutes)."
          .format(elapsed, elapsed / 60))

    k_feat = [len(k) for k in sbs.subsets_]

    f, ax = plt.subplots(1, figsize=(width, height))
    plt.plot(k_feat, sbs.scores_, marker='o')
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    if output == 'show':
        plt.show()
    elif output == 'save':
        f.savefig(save_path, dpi=300, bbox_inches='tight')
    if return_feats:
        return sbs.subsets_


def fit_rfecv(classifier, X, y, model_name,
              step=1, kfold=2, fig_width=6, fig_height=4):
    t = time()

    rfecv = RFECV(estimator=classifier, step=step, cv=StratifiedKFold(kfold),
                  scoring='accuracy')
    rfecv.fit(X, y)

    print("{0} fit using RFE".format(model_name))
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    f, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    elapsed = time() - t
    print("RFE fit, took, {0:,.2f} seconds ({1:,.2f} minutes)".format(elapsed, elapsed / 60))


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
              return_coefs=False, return_scores=False, verbose=True,
              feat_names=None, class_names=None, plot_dec_reg=None):
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
    if verbose:
        print("\n{0} fit, took {1:,.2f} seconds ({2:,.2f} minutes)".format(model_name, elapsed, elapsed / 60) +
              "\naccuracy: train={0:.2f}, test={1:.2f}, validation #1={2:.2f}, validation #2={3:.2f}"
              .format(train_score, test_score, val1_score, val2_score))

    if plot_dec_reg == 'train-test':
        X_combined = np.vstack((X_train, X_test))
        y_combined = np.hstack((y_train, y_test))
        plot_decision_regions(X=X_combined,
                              y=y_combined,
                              classifier=model)
    elif plot_dec_reg == 'val1':
        plot_decision_regions(X=X_val1,
                              y=y_val1,
                              classifier=model)
    elif plot_dec_reg == 'val2':
        plot_decision_regions(X=X_val2,
                              y=y_val2,
                              classifier=model)

    if return_coefs:
        if feat_names is None:
            feat_names = range(X_train.shape[1])
        if class_names is None:
            class_names = range(model.coef_.shape[0])
        coef_df = pd.DataFrame()
        for cl in range(model.coef_.shape[0]):
            class_coef = pd.DataFrame(model.coef_[cl], index=feat_names).reset_index() \
                .rename(columns={'index': 'var', 0: 'coefficient'})
            class_coef['class'] = class_names[cl]
            coef_df = coef_df.append(class_coef)
        return coef_df
    elif return_scores:
        return train_score, test_score, val1_score, val2_score


def targets_corr(df, target_list, target_var, plot_corr=True, print_top_coefs=True, print_top=10,
                 fig_height=4, fig_width=10, legend_loc='center right',
                 output='show', save_path='targets_corr.png', dpi=300, save_only=False):
    target0_corr = df.corr()[target_list[0]].reset_index().rename(columns={'index': 'var', 'variable': 'class'})
    target1_corr = df.corr()[target_list[1]].reset_index().rename(columns={'index': 'var', 'variable': 'class'})
    target2_corr = df.corr()[target_list[2]].reset_index().rename(columns={'index': 'var', 'variable': 'class'})
    target3_corr = df.corr()[target_list[3]].reset_index().rename(columns={'index': 'var', 'variable': 'class'})

    all_targets_corr = pd.merge(
        pd.merge(
            pd.merge(target0_corr, target1_corr, on='var'),
            target2_corr, on='var'),
        target3_corr, on='var')
    target_list.append(target_var)
    mask1 = all_targets_corr['var'].isin(target_list)
    all_targets_corr = all_targets_corr[~mask1]
    targets_corr_tidy = pd.melt(all_targets_corr, id_vars='var').sort_values('var')

    if print_top_coefs:
        print("----- Pearson correlation coefficient between features and target classes"
              "\n\n         strongest negative correlation (top {0}):\n".format(print_top),
              targets_corr_tidy.sort_values('value').head(print_top),
              "\n\n         strongest positive correlation (top {0}):\n".format(print_top),
              targets_corr_tidy.sort_values('value', ascending=False).head(print_top))

    if plot_corr:
        # plot univariate Pearson correlation coefficients with target classes
        f, ax = plt.subplots(1, figsize=(fig_width, fig_height))
        sns.barplot(x="value", y="var", hue="variable", data=targets_corr_tidy,
                    palette="muted", ax=ax)
        ax.set_ylabel("Features", fontsize=16)
        ax.set_xlabel("Correlation coefficient", fontsize=16)
        ax.set_title("Pearson correlation coefficient between features and target classes", fontsize=16)
        ax.grid(True)
        ax.legend(loc=legend_loc, fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if output == 'show':
            plt.show()
        if output == 'save':
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            if save_only:
                f.close()


def targets_corr_3c(df, target_list, target_var, plot_corr=True, print_top_coefs=True, print_top=10,
                    fig_height=4, fig_width=10, legend_loc='center right',
                    output='show', save_path='targets_corr.png', dpi=300, save_only=False):

    target0_corr = df.corr()[target_list[0]].reset_index().rename(columns={'index': 'var', 'variable': 'class'})
    target1_corr = df.corr()[target_list[1]].reset_index().rename(columns={'index': 'var', 'variable': 'class'})
    target2_corr = df.corr()[target_list[2]].reset_index().rename(columns={'index': 'var', 'variable': 'class'})

    all_targets_corr = pd.merge(
        pd.merge(target0_corr, target1_corr, on='var'),
        target2_corr, on='var')

    target_list.append(target_var)
    mask1 = all_targets_corr['var'].isin(target_list)
    all_targets_corr = all_targets_corr[~mask1]
    targets_corr_tidy = pd.melt(all_targets_corr, id_vars='var').sort_values('var')

    if print_top_coefs:
        print("----- Pearson correlation coefficient between features and target classes"
              "\n\n         strongest negative correlation (top {0}):\n".format(print_top),
              targets_corr_tidy.sort_values('value').head(print_top),
              "\n\n         strongest positive correlation (top {0}):\n".format(print_top),
              targets_corr_tidy.sort_values('value', ascending=False).head(print_top))

    if plot_corr:
        # plot univariate Pearson correlation coefficients with target classes
        f, ax = plt.subplots(1, figsize=(fig_width, fig_height))
        sns.barplot(x="value", y="var", hue="variable", data=targets_corr_tidy,
                    palette="muted", ax=ax)
        ax.set_ylabel("Features", fontsize=16)
        ax.set_xlabel("Correlation coefficient", fontsize=16)
        ax.set_title("Pearson correlation coefficient between features and target classes", fontsize=16)
        ax.grid(True)
        ax.legend(loc=legend_loc, fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if output == 'show':
            plt.show()
        if output == 'save':
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            if save_only:
                f.close()


def plot_learning_curve(classifier, model_name, X, y, n_jobs=1, cv=10, num_train_sizes=10,
                        fig_height=4, fig_width=6):
    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
                                                            X=X,
                                                            y=y,
                                                            train_sizes=np.linspace(0.1, 1.0, num_train_sizes),
                                                            cv=cv,
                                                            n_jobs=n_jobs)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    f, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Accuracy')
    ax.set_title("{0}, learning curve".format(model_name))
    plt.legend(loc='lower right')

    plt.show()


def plot_validation_curve(classifier, model_name, X, y, param_name, param_range, n_jobs=1, cv=10,
                          fig_height=4, fig_width=6, xlog=False):
    t = time()

    train_scores, test_scores = validation_curve(estimator=classifier, X=X, y=y, cv=cv, n_jobs=n_jobs,
                                                 param_name=param_name, param_range=param_range)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    f, ax = plt.subplots(1, figsize=(fig_width, fig_height))

    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.legend(loc='lower right')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    ax.set_title("{0}, validation curve\n{1}: {2}".format(model_name, param_name,
                                                          list(pd.Series(param_range).apply(lambda x: round(x, 3)))))
    plt.legend(loc='lower right')
    if xlog:
        plt.xscale('log')

    elapsed = time() - t
    print("Validation curve for {0} plotted, took {1:,.2f} seconds ({2:,.2f} minutes)"
          .format(model_name, elapsed, elapsed / 60))
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, model_name,
                          normalize=False,
                          interpolation='nearest',
                          cmap=plt.cm.Blues,
                          width=4):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        title = '{0}\nNormalized confusion matrix'.format(model_name)
    else:
        title = '{0}\nConfusion matrix, without normalization'.format(model_name)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("{0}\nNormalized confusion matrix".format(model_name))
    else:
        print('{0}\nConfusion matrix, without normalization'.format(model_name))

    print(cm)

    fig, ax = plt.subplots()
    plt.grid(False)
    fig.set_size_inches((width, width))

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    ax.tick_params(labelsize=16)
    ax.set_xlabel('Predicted label', fontsize=18)
    ax.set_ylabel('True label', fontsize=18)
    ax.set_title(title, fontsize=18)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else ',d'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)
    fig.tight_layout()
    return ax


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


# dot_data = export_graphviz(model, filled=True, rounded=True,
#                            class_names=['def', 'for'],
#                            feature_names=[xcol1, xcol2],
#                            out_file=None)
# graph = graph_from_dot_data(dot_data)
# graph.write_png('img/tree.png')
