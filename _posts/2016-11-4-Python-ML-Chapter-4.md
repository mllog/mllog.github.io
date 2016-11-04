---
layout: post
title: Python Machine Learning - Chapter 4
subtitle: Building Good Training Sets – Data Preprocessing
tags: [machine learning, kaggle]
---

# Topics

- Removing and imputing missing values from the dataset
- Getting categorical data into shape for machine learning algorithms
- Selecting relevant features for the model construction

# 1. Dealing with missing data

Create a simple example dataframe from CSV file


```python
import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''
```


```python
df = pd.read_csv(StringIO(csv_data))
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Couting number of missing values


```python
df.isnull().sum()
```




    A    0
    B    0
    C    1
    D    1
    dtype: int64



### Eliminating samples or features with missing values

Remove rows with missing values


```python
df.dropna()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



Remove columns with missing values


```python
df.dropna(axis=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>



The `dropna` method supports several additional parameters


```python
# only drop rows where all columns are NaN
df.dropna(how='all')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# only drop rows where NaN appear in specific columns
df.dropna(subset=['C'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Imputing missing values

We can replace missing values with more meaningfull values such as mean value of feature, most-frequent values, median


```python
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
```


```python
imr = imr.fit(df)
```


```python
imputed_data = imr.transform(df.values)
imputed_data
```




    array([[  1. ,   2. ,   3. ,   4. ],
           [  5. ,   6. ,   7.5,   8. ],
           [  0. ,  11. ,  12. ,   6. ]])



# 1. Handling categorical data

Real-world datas often contain categorical feature columns. There are two types of categorical data: **nominal** amd **ordinal**. Ordinal features can be understood as categorical values that can be sorted or ordered. In contrast, mominal features don't imply any order. 


```python
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                  ['red', 'L', 13.5, 'class2'],
                  ['blue', 'XL', 15.5, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>classlabel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>M</td>
      <td>10.1</td>
      <td>class1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>L</td>
      <td>13.5</td>
      <td>class2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>blue</td>
      <td>XL</td>
      <td>15.5</td>
      <td>class1</td>
    </tr>
  </tbody>
</table>
</div>



As we can see in the preceding output, the newly created DataFrame contains a nominal feature (color), an ordinal feature (size), and a numerical feature (price) column. The class labels (assuming that we created a dataset for a supervised learning task) are stored in the last column. The learning algorithms for classification that we discuss in this book do not use ordinal information in class labels.

### Mapping ordinal features

To make sure that the learning algorithm interprets the ordinal features correctly, we need to convert the categorical string values into integers. Unfortunately, there is no convenient function that can automatically derive the correct order of the labels of our size feature. Thus, we have to define the mapping manually.


```python
size_mapping = {'M':1, 'L':2, 'XL':3}
df['size'] = df['size'].map(size_mapping)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>classlabel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>1</td>
      <td>10.1</td>
      <td>class1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>2</td>
      <td>13.5</td>
      <td>class2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>blue</td>
      <td>3</td>
      <td>15.5</td>
      <td>class1</td>
    </tr>
  </tbody>
</table>
</div>



### Encoding class lablels

Many machine learning libraries require that class labels are encoded as integer values. Although most estimators for classification in scikit-learn convert class labels to integers internally, it is considered good practice to provide class labels as integer arrays to avoid technical glitches.


```python
import numpy as np
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping
```




    {'class1': 0, 'class2': 1}




```python
df['classlabel'] = df['classlabel'].map(class_mapping)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>classlabel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>1</td>
      <td>10.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>2</td>
      <td>13.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>blue</td>
      <td>3</td>
      <td>15.5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We can reverse the key-value pairs in the mapping dictionary as follows to map the converted class labels back to the original string representation:


```python
inv_class_mapping = {v:k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>classlabel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>1</td>
      <td>10.1</td>
      <td>class1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>2</td>
      <td>13.5</td>
      <td>class2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>blue</td>
      <td>3</td>
      <td>15.5</td>
      <td>class1</td>
    </tr>
  </tbody>
</table>
</div>



Alternatively, there is a convenient LabelEncoder class directly implemented in scikit-learn to achieve the same:


```python
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
```


```python
y = class_le.fit_transform(df['classlabel'].values)
y
```




    array([0, 1, 0])



Note that the fit_transform method is just a shortcut for calling fit and transform separately, and we can use the inverse_transform method to
transform the integer class labels back into their original string representation:


```python
class_le.inverse_transform(y)
```




    array(['class1', 'class2', 'class1'], dtype=object)



### Performing one-hot encoding on nominal features

In the previous section, we used a simple dictionary-mapping approach to convert the ordinal size feature into integers. Since scikit-learn's estimators treat class labels without any order, we used the convenient LabelEncoder class to encode the string labels into integers. It may appear that we could use a similar approach to transform the nominal color column of our dataset, as follows:


```python
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X
```




    array([['green', 1, 10.1],
           ['red', 2, 13.5],
           ['blue', 3, 15.5]], dtype=object)




```python
X[:,0] = color_le.fit_transform(X[:,0])
X
```




    array([[1, 1, 10.1],
           [2, 2, 13.5],
           [0, 3, 15.5]], dtype=object)



After executing the preceding code, the first column of the NumPy array X now holds the new color values, which are encoded as follows:
- blue -> 0
- green -> 1
- red ->2

If we stop at this point and feed the array to our classifier, we will make one of the most common mistakes in dealing with categorical data. Can you spot the problem? Although the color values don't come in any particular order, a learning algorithm will now assume that green is larger than blue, and red is larger than green. Although this assumption is incorrect, the algorithm could still produce useful results. However, those results would not be optimal.

A common workaround for this problem is to use a technique called one-hot encoding. The idea behind this approach is to create a new dummy feature for each unique value in the nominal feature column. Here, we would convert the color feature into three new features: blue, green, and red. Binary values can then be used to indicate the particular color of a sample; for example, a blue sample can be encoded as blue=1, green=0, red=0. To perform this transformation, we can use the OneHotEncoder that is implemented in the scikit-learn.preprocessing module:


```python
from sklearn.preprocessing import OneHotEncoder
```


```python
ohe = OneHotEncoder(categorical_features=[0], sparse=False)
```


```python
ohe.fit_transform(X)
```




    array([[  0. ,   1. ,   0. ,   1. ,  10.1],
           [  0. ,   0. ,   1. ,   2. ,  13.5],
           [  1. ,   0. ,   0. ,   3. ,  15.5]])




```python
pd.get_dummies(df[['price', 'color', 'size']])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>size</th>
      <th>color_blue</th>
      <th>color_green</th>
      <th>color_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.1</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.5</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.5</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Partitioning a dataset in training and test sets


```python
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
```


```python
print('Class label', np.unique(df_wine['Class label']))
```

    Class label [1 2 3]



```python
df_wine.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class label</th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_wine.shape
```




    (178, 14)




```python
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

### Bringing features onto the same scale

Feature scaling is a crucial step in our preprocessing pipeline that can easily be forgotten. Decision trees and random forests are one of the very few machine learning algorithms where we don't need to worry about feature scaling. However, the majority of machine learning and optimization algorithms behave much
better if features are on the same scale.

There are two common approaches to bringing different features onto the same scale: normalization and standardization.

To normalize our data, we can simply apply the min-max scaling to each feature column, where the new value $x_{norm}^{(i)}$ of a sample $x^{(i)}$ can be calculated as follows:

$$x_{norm}^{(i)} = \frac{x^{(i)} - x_{min}}{x_{max} - x_{min}}$$


```python
from sklearn.preprocessing import MinMaxScaler
mns = MinMaxScaler()
```


```python
X_train_norm = mns.fit_transform(X_train)
X_test_norm = mns.fit_transform(X_test)
```

Although normalization via min-max scaling is a commonly used technique that is useful when we need values in a bounded interval, standardization can be more practical for many machine learning algorithms.

The procedure of standardization can be expressed by the following equation:

$$x_{std}^{(i)} = \frac{x^{(i)} - \mu_x}{\sigma_x}$$

Here, $\mu_x$ is the sample mean of a particular feature column and $\sigma_x$ the corresponding standard deviation, respectively.


```python
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
```


```python
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)
```

### Selecting meaningful features

If we notice that a model performs much better on a training dataset than on the test dataset, this observation is a strong indicator for overfitting. Overfitting means that model fits the parameters too closely to the particular observations in the training dataset but does not generalize well to real data—we say that the model has a high variance. A reason for overfitting is that our model is too complex for the given training data and common solutions to reduce the generalization error are listed
as follows:
- Collect more training data
- Introduce a penalty for complexity via regularization
- Choose a simpler model with fewer parameters
- Reduce the dimensionality of the data

Common ways to reduce overfitting by regularization and dimensionality reduction via feature selection.

#### Sparse solutions with L1 regularization

For regularized models in scikit-learn that support L1 regularization, we can simply set the penalty parameter to 'l1' to yield the sparse solution:


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.1)
```


```python
lr.fit(X_train_std, y_train)
print('Training Accuracy: ', lr.score(X_train_std, y_train))
```

    Training Accuracy:  0.983870967742



```python
print('Testing Accuracy: ', lr.score(X_test_std, y_test))
```

    Testing Accuracy:  0.981481481481



```python
lr.intercept_
```




    array([-0.38382541, -0.15814486, -0.70038079])




```python
lr.coef_
```




    array([[ 0.28019587,  0.        ,  0.        , -0.0280669 ,  0.        ,
             0.        ,  0.70999026,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  1.23632608],
           [-0.64374975, -0.068967  , -0.05718861,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        , -0.9271808 ,
             0.05973745,  0.        , -0.37096885],
           [ 0.        ,  0.06139606,  0.        ,  0.        ,  0.        ,
             0.        , -0.63693551,  0.        ,  0.        ,  0.49844523,
            -0.35805222, -0.57047635,  0.        ]])



We notice that the weight vectors are sparse, which means that they only have a few non-zero entries. As a result of the L1 regularization, which serves as a method for feature selection, we just trained a model that is robust to the potentially irrelevant features in this dataset.


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
def regularized_paths():
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
    weights, params = [], []
    for c in range(-4, 6):
        lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)
    weights = np.array(weights)
    for col, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:,col], label=df_wine.columns[col+1], color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
    plt.show()
```


```python
regularized_paths()
```


![png](output_79_0.png)


The resulting plot provides us with further insights about the behavior of L1 regularization. As we can see, all features weights will be zero if we penalize the model with a strong regularization parameter (C < 0.1); C is the inverse of the regularization parameter $\lambda$.

#### Assessing feature importance with random forests


```python
from sklearn.ensemble import RandomForestClassifier
feat_lables = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_lables[f], importances[indices[f]]))
```

     1) Alcohol                        0.182483
     2) Malic acid                     0.158610
     3) Ash                            0.150948
     4) Alcalinity of ash              0.131987
     5) Magnesium                      0.106589
     6) Total phenols                  0.078243
     7) Flavanoids                     0.060718
     8) Nonflavanoid phenols           0.032033
     9) Proanthocyanins                0.025400
    10) Color intensity                0.022351
    11) Hue                            0.022078
    12) OD280/OD315 of diluted wines   0.014645
    13) Proline                        0.013916



```python
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_lables, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
```


![png](output_83_0.png)


we could set the threshold to 0.15 to reduce the dataset to the 3 most important features, Alcohol, Malic acid, and Ash using the following code:


```python
X_selected = forest.transform(X_train, threshold=0.15)
X_selected.shape
```

    /Users/trang/anaconda2/envs/py35/lib/python3.5/site-packages/sklearn/utils/__init__.py:93: DeprecationWarning: Function transform is deprecated; Support to use estimators as feature selectors will be removed in version 0.19. Use SelectFromModel instead.
      warnings.warn(msg, category=DeprecationWarning)





    (124, 3)




```python

```
