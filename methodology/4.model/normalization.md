[GitHub Flavored Markdown Spec](https://github.github.com/gfm/)
# Description of methodology
# Rescaling variables
From [machinelearningmastery.com](https://machinelearningmastery.com/normalize-standardize-time-series-data-python/), [wikipedia](https://en.wikipedia.org/wiki/Feature_scaling), and a [lecture by Andrew Ng](http://openclassroom.stanford.edu/MainFolder/VideoPage.php?course=MachineLearning&video=03.1-LinearRegressionII-FeatureScaling&speed=100/) on Feature Scaling:

Some machine learning algorithms will achieve better performance if data has a consistent scale or distribution. Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization.

For example, the majority of classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it. In stochastic gradient descent, feature scaling can sometimes improve the convergence speed of the algorithm. In support vector machines, it can reduce the time to find support vectors. Note that feature scaling changes the SVM result.

Two techniques that can be used to consistently rescale data are :
### Normalization 
* Also known as feature scaling or unity-based normalization
* Normalization is a rescaling of the data from the original range so that all values are within the range of 0 and 1.
* Normalization can be useful, and even required in some machine learning algorithms when data has input values with differing scales.
* It may be required for algorithms, like k-Nearest neighbors, which uses distance calculations and Linear Regression and Artificial Neural Networks that weight input values.
* Normalization requires the knowledge or accurate estimation of the minimum and maximum observable values (can be estimated from the available data).
* If needed, the transform can be inverted. This is useful for converting predictions back into their original scale for reporting or plotting.
* If the data presents a time series that is trending up or down, estimating these expected values may be difficult and normalization may not be the best method to use.
* Types of normalization:
    * Rescaling (min-max normalization)  
$ \large{ X' = \frac{ X - X_{min} } { X_{max} - X_{min} } } $
    * Rescaling between an arbitrary set of values  
$ \large{ X' = a + \frac{ (X - X_{min})(b - a) } { X_{max} - X_{min} } } $
    * Mean normalization  
$ \large{ X' = \frac{ X - \mu_X } { X_{max} - X_{min} } } $

Variables can be normalized using the `scikit-learn` object `MinMaxScaler`.
### Standardization (Z-score normalization)
* Standardization is another type of rescaling that is more robust to new values being outside the range of expected values than normalization. 
*  Feature standardization makes the values of each feature in the data have zero-mean (when subtracting the mean in the numerator) and unit-variance.
    * This can be thought of as subtracting the mean value, or centering the data, and scaling by standard deviation.
* Like normalization, standardization can be useful, and even required in some machine learning algorithms when data has input values with differing scales.
    * This method is widely used for normalization in many machine learning algorithms (e.g., support vector machines, logistic regression, and artificial neural networks).
* Standardization assumes that observations fit a [Gaussian distribution](http://hyperphysics.phy-astr.gsu.edu/hbase/Math/gaufcn.html) (bell curve) with a well behaved mean and standard deviation. 
    * Data can still be standardized if this expectation is not met, but results might not be reliable.
* Standardization requires the knowledge or accurate estimation of the mean and standard deviation of observable values. 
    * These values can be estimated from training data.
* Types of standardization
    * General standardization  
$ \large{ X' = \frac{ X - \mu_X } { \sigma } } $,
where $\mu_X$ is the mean of the feature and $\sigma$ is its standard deviation
    * Scaling to unit length
        * Another option that is widely used in machine-learning is to scale the components of a feature vector such that the complete vector has length one. 
        * This usually means dividing each component by the [Euclidean length](https://en.wikipedia.org/wiki/Euclidean_length) of the vector:  
$ \large{ X' = \frac{ X } { ||X||_2 } } $
        * In some applications (e.g. Histogram features) it can be more practical to use the L1 norm (i.e. Manhattan Distance, City-Block Length or [Taxicab Geometry](https://en.wikipedia.org/wiki/Taxicab_Geometry)) of the feature vector. 
        * This is especially important if in the following learning steps the Scalar Metric is used as a distance measure.

Variables can be standardized using the `scikit-learn` object `StandardScaler`.
