---
layout: post
title: Python Machine Learning - Chapter 3
subtitle: A Tour of ML Classifier Using Scikit-learn
tags: [machine learning, kaggle]
---


# Choosing a classification algorithm


## The five main steps that are involved in training a ML algorithm
1. Selection of features
2. Choosing a performance metric
3. Choosing a classifier and optimization algorithm
4. Evaluating the performance of the model
5. Tuning the algorithm

## First steps with scikit-learn

### Training a perceptron via scikit-learn


```python
from sklearn import datasets
import numpy as np
```


```python
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
```


```python
from sklearn.cross_validation import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

#### Feature scaling: standardize the features using StandardScaler


```python
from sklearn.preprocessing import StandardScaler
```


```python
sc = StandardScaler()
sc.fit(X_train)
```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```python
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

#### Train a perceptron model: Most algorithms in scikit-learn already support multiclass classification by default via the 0ne-vs.-Rest method


```python
from sklearn.linear_model import Perceptron
```


```python
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
```




    Perceptron(alpha=0.0001, class_weight=None, eta0=0.1, fit_intercept=True,
          n_iter=40, n_jobs=1, penalty=None, random_state=0, shuffle=True,
          verbose=0, warm_start=False)




```python
y_pred = ppn.predict(X_test_std)
```


```python
print('Misclassfified samples: %d' % (y_test != y_pred).sum())
```

    Misclassfified samples: 4


#### Instead of using misclassification error, we can use accuracy


```python
from sklearn.metrics import accuracy_score
```


```python
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
```

    Accuracy: 0.91



```python
%matplotlib inline
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
```


```python
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')
    
```


```python
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_std = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined_std, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_22_0.png)


#### Perceptron algorithm never converges on datasets that aren't perfectly linearly separable, which is why the use of perceptron algorithm is typically not recommended in practice.

### Modeling class probabilities via logistic regression

#### Activation function is a *logistic* function or *sigmoid* function:
#### $$\phi(z) = \frac{1}{1 + e^{-z}} \\
z = w^Tx$$
![Logistic regression](https://mllog.github.io/img/pythonmlch2/logisticregression.png)

#### Example of sigmoid function


```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
```


```python
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_27_0.png)


### Training a logistic regression model with scikit-learn
#### With logistic regression, the cost function is:
#### $$J(w) = - \sum_i y^{(i)}log(\phi (z^{(i)})) + (1 - y^{(i)})log(1 - \phi (z^{(i)}))$$


```python
from sklearn.linear_model import LogisticRegression
```


```python
lr = LogisticRegression(C=1000.0, random_state=0)
```


```python
lr.fit(X_train_std, y_train)
```




    LogisticRegression(C=1000.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=0,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)




```python
plot_decision_regions(X=X_combined_std, y=y_combined_std, classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_32_0.png)



```python
lr.predict_proba(X_test_std[0,:])
```

    /Users/trang/anaconda2/envs/py35/lib/python3.5/site-packages/sklearn/utils/validation.py:386: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      DeprecationWarning)





    array([[  2.05743774e-11,   6.31620264e-02,   9.36837974e-01]])



### What is the C parameter in the LogisticRegression?
#### In order to deal with overfitting problem, we use regularization. The most common form of regularization is the so-call **L2 regularization** written as follows:
#### $$\frac{\lambda}{2}\lVert x \rVert ^2 = \frac{\lambda}{2}\sum\limits_{j=1}^m w_j^2$$
#### To apply regularization, we need to add the regularization term to the cost function:
#### $$J(w) = \left[\sum_i y^{(i)}(-log(\phi (z^{(i)}))) + (1 - y^{(i)})(-log(1 - \phi (z^{(i)})))\right] + \frac{\lambda}{2}\lVert x \rVert ^2 $$
#### The parameter $C$ that is implemented for the LogisticRegression class in scikit-learn is inverse of $\lambda$
#### $$C = \frac{1}{\lambda}$$


## Maximum margin classification with Support Vector Machines (SVM)
### With SVMs optimization objective is to maximize the **margin**. The margin is defined as the distance between the separating hyperplane (decision boundary) and the training samples that are closest to this boundary. This is illustrated in the following figure:
![SVM](https://mllog.github.io/img/pythonmlch2/svm.png)


```python
from sklearn.svm import SVC
```


```python
svm = SVC(kernel='linear', C=1.0, random_state=0)
```


```python
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined_std, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petan width [standardized]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_38_0.png)


### Logistic regression versus SVM
#### In practical classification tasks, linear logistic regression and linear SVMs often yield very similar results. Logistic regression tries to maximize the conditional likelihoods of the training data, which makes it more prone to outliers than SVMs. The SVMs mostly care about the points that are closest to the decision boundary (support vectors). On the other hand, logistic regression has the advantage that it is a simpler model that can be implemented more easily. Furthermore, logistic regression models can be easily updated, which is attractive when working with streaming data.

### Alternative implemtations for perceptron, logistic regression and SVM in scikit-learn
#### The advantage of using LIBLINEAR and LIBSVM over native Python implementations is that they allow an extremely quick training of large amounts of linear classifiers. However, sometimes our datasets are too large to fit into computer memory. Thus, scikit-learn also offers alternative implementations via the SGDClassifier class, which also supports online learning via the partial_fit method. The concept behind the SGDClassifier class is similar to the stochastic gradient algorithm


```python
from sklearn.linear_model import SGDClassifier
```


```python
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')
```


```python
ppn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined_std, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petan width [standardized]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_43_0.png)



```python
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined_std, classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petan width [standardized]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_44_0.png)



```python
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined_std, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petan width [standardized]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_45_0.png)


## Solving nonlinear problems using a kernel SVM
### Another reason why SVMs enjoy high popularity among machine learning practitioners is that they can be easily kernelized to solve nonlinear classification problems.
#### One of the most widely used kernels is the **Radial Basis Function kernel (RBF)** or Gaussian kernel:
#### $$k(x^{(i)}, x^{(j)}) = exp(-\frac{\lVert x^{(i)} - x^{(j)} \rVert ^2}{2\sigma ^2})$$
#### This is often simplified to: $$k(x^{(i)}, x^{(j)}) = exp(-\gamma \lVert x^{(i)} - x^{(j)} \rVert ^2) \\ \gamma = \frac{1}{2\sigma ^ 2}$$


```python
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_47_0.png)



```python
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_48_0.png)


#### The  parameter, which we set to *gamma=0.1*, can be understood as a cut-off parameter for the Gaussian sphere. If we increase the value for $\gamma$, we increase the influence or reach of the training samples, which leads to a softer decision boundary. To get a better intuition for $\gamma$, let's apply RBF kernel SVM to our Iris flower dataset:


```python
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined_std, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petan width [standardized]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_50_0.png)


#### Since we chose a relatively small value for $\gamma$, the resulting decision boundary of the RBF kernel SVM model will be relatively soft. Now let's increase the value of $\gamma$ and see what happeds


```python
svm = SVC(kernel='rbf', random_state=0, gamma=100, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined_std, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petan width [standardized]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_52_0.png)


#### In the resulting plot, we can now see that the decision boundary around the classes 0 and 1 is much tighter. Although the model fits the training dataset very well, such a classifier will likely have a high generalization error on unseen data, which illustrates that the optimization of $\gamma$  also plays an important role in controlling overfitting.

## Decision tree learning
### Decision tree classifiers are attractive models if we care about interpretability
### Objective function to split the nodes at the most informative features:
#### $$IG(D_p, f) = I(D_p) - \sum_{j=1}^{m}\frac{N_j}{N_p}I(D_j)$$
#### $f$ is the feature to perform the split, $D_p$ and $D_j$ are the dataset of the parent and $jth$ child node, I is our impurity measure, $N_p$ is the total number of samples at the parent node, and $N_j$ is the number of samples in the $jth$ child node.
### Most libraries implement binary decision trees:
#### Objective function: $$IG(D_p, a) = I(D_p) - \frac{N_{left}}{N_p}I(D_{left}) -  \frac{N_{right}}{N_p}I(D_{right})$$
Three commonly used impurity measures in binary decision trees are: **Gini index ($I_G$)**, **entropy ($I_H$)**, and the **classification error ($I_E$)**


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
def gini(p):
    return (p)*(1 - (p)) + (1-p)*(1-(1-p))
def entropy(p):
    return - p*np.log2(p) - (1-p)*np.log2((1-p))
def error(p):
    return 1 - np.max([p, 1-p])
```


```python
x= np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c in zip([ent, sc_ent, gini(x), err], ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'], 
                        ['-', '-', '--', '-.'],
                        ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_56_0.png)


### Building a decision tree


```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
```


```python
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_59_0.png)



```python
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])
```

## Combining weak to strong learners via random forests
### a random forest can be considered as an ensemble of decision trees. The idea behind ensemble learning is to combine weak learners to build a more robust model that has a better generalization error and is less susceptible to overfitting. Four simple steps of random forest algirithm:
1. Draw a random **bootstrap** sample of size n
2. Grow a decision tree from the bootstrap sample. At each node:
    1. Randomly select $d$ features without replacement
    2. Split the node using the feature that provides the best split according to the objective function
3. Repeat the steps 1 to 2 k times
4. Aggregate the prediction by each tree to assign the class lable by **majority vote**.


```python
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
```


```python
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_63_0.png)


## K-nearest neighbors - a lazy learning
### KNN algorithm:
1. Choose the number of $k$ and a distance metric
2. Find the $k$ nearest neighbors of the sample that we want to classify
3. Assign the class lable by majority vote.


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
```


```python
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.show()
```


![png](https://mllog.github.io/img/pythonmlch3/output_66_0.png)



```python

```
