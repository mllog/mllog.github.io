---
layout: post
title: Exploratory Study on ML Algorithms
subtitle: A Notebook for Alstate security competition
tags: [machine learning, Hand-on]
---


# This is a Notebook for Alstate competition.

The problem is to predict the 'loss' based on the other attributes. So, this is a regression problem



```python
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

import pandas
```


```python
dataset = pandas.read_csv("input/train.csv")
dataset_test = pandas.read_csv("input/test.csv")
ID = dataset_test['id']
dataset_test.drop('id', axis=1, inplace=True)

#Print all rows and columns
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
```


```python
print(dataset.shape)
```

    (188318, 132)



```python
dataset = dataset.iloc[:,1:]
```


```python
dataset.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cont1</th>
      <th>cont2</th>
      <th>cont3</th>
      <th>cont4</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.493861</td>
      <td>0.507188</td>
      <td>0.498918</td>
      <td>0.491812</td>
      <td>0.487428</td>
      <td>0.490945</td>
      <td>0.484970</td>
      <td>0.486437</td>
      <td>0.485506</td>
      <td>0.498066</td>
      <td>0.493511</td>
      <td>0.493150</td>
      <td>0.493138</td>
      <td>0.495717</td>
      <td>3037.337686</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.187640</td>
      <td>0.207202</td>
      <td>0.202105</td>
      <td>0.211292</td>
      <td>0.209027</td>
      <td>0.205273</td>
      <td>0.178450</td>
      <td>0.199370</td>
      <td>0.181660</td>
      <td>0.185877</td>
      <td>0.209737</td>
      <td>0.209427</td>
      <td>0.212777</td>
      <td>0.222488</td>
      <td>2904.086186</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000016</td>
      <td>0.001149</td>
      <td>0.002634</td>
      <td>0.176921</td>
      <td>0.281143</td>
      <td>0.012683</td>
      <td>0.069503</td>
      <td>0.236880</td>
      <td>0.000080</td>
      <td>0.000000</td>
      <td>0.035321</td>
      <td>0.036232</td>
      <td>0.000228</td>
      <td>0.179722</td>
      <td>0.670000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.346090</td>
      <td>0.358319</td>
      <td>0.336963</td>
      <td>0.327354</td>
      <td>0.281143</td>
      <td>0.336105</td>
      <td>0.350175</td>
      <td>0.312800</td>
      <td>0.358970</td>
      <td>0.364580</td>
      <td>0.310961</td>
      <td>0.311661</td>
      <td>0.315758</td>
      <td>0.294610</td>
      <td>1204.460000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.475784</td>
      <td>0.555782</td>
      <td>0.527991</td>
      <td>0.452887</td>
      <td>0.422268</td>
      <td>0.440945</td>
      <td>0.438285</td>
      <td>0.441060</td>
      <td>0.441450</td>
      <td>0.461190</td>
      <td>0.457203</td>
      <td>0.462286</td>
      <td>0.363547</td>
      <td>0.407403</td>
      <td>2115.570000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.623912</td>
      <td>0.681761</td>
      <td>0.634224</td>
      <td>0.652072</td>
      <td>0.643315</td>
      <td>0.655021</td>
      <td>0.591045</td>
      <td>0.623580</td>
      <td>0.566820</td>
      <td>0.614590</td>
      <td>0.678924</td>
      <td>0.675759</td>
      <td>0.689974</td>
      <td>0.724623</td>
      <td>3864.045000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.984975</td>
      <td>0.862654</td>
      <td>0.944251</td>
      <td>0.954297</td>
      <td>0.983674</td>
      <td>0.997162</td>
      <td>1.000000</td>
      <td>0.980200</td>
      <td>0.995400</td>
      <td>0.994980</td>
      <td>0.998742</td>
      <td>0.998484</td>
      <td>0.988494</td>
      <td>0.844848</td>
      <td>121012.250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.skew()
```




    cont1     0.516424
    cont2    -0.310941
    cont3    -0.010002
    cont4     0.416096
    cont5     0.681622
    cont6     0.461214
    cont7     0.826053
    cont8     0.676634
    cont9     1.072429
    cont10    0.355001
    cont11    0.280821
    cont12    0.291992
    cont13    0.380742
    cont14    0.248674
    loss      3.794958
    dtype: float64



## Data Visualization (Skip)


```python
import numpy
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
```

## Data transformation
- Skew correction


```python
dataset["loss"] = numpy.log1p(dataset["loss"])
sns.violinplot(data=dataset, y="loss")
plt.show()
```


![png](https://mllog.github.io/img/output_10_0.png)


## Data Interaction
- Correlation


```python
# Correlation tells relation between two attributes.
# Correlation requires continous data. Hence, ignore categorical data

# Calculates pearson co-efficient for all combinations
#create a dataframe with only continuous features
#range of features considered
split = 116 

#number of features considered
size = 15

data=dataset.iloc[:,split:] 

cols=data.columns 

data_corr = data.corr()
threshold = 0.5
corr_list = []

for i in range(0,size):
    for j in range(i+1,size):
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j])
            
s_corr_list = sorted(corr_list, key=lambda x: -abs(x[0]))

for v,i,j in s_corr_list:
    print("%s and %s = %.2f" %(cols[i], cols[j], v))
```

    cont11 and cont12 = 0.99
    cont1 and cont9 = 0.93
    cont6 and cont10 = 0.88
    cont6 and cont13 = 0.82
    cont1 and cont10 = 0.81
    cont6 and cont9 = 0.80
    cont9 and cont10 = 0.79
    cont6 and cont12 = 0.79
    cont6 and cont11 = 0.77
    cont1 and cont6 = 0.76
    cont7 and cont11 = 0.75
    cont7 and cont12 = 0.74
    cont10 and cont12 = 0.71
    cont10 and cont13 = 0.71
    cont10 and cont11 = 0.70
    cont6 and cont7 = 0.66
    cont9 and cont13 = 0.64
    cont9 and cont12 = 0.63
    cont1 and cont12 = 0.61
    cont9 and cont11 = 0.61
    cont1 and cont11 = 0.60
    cont1 and cont13 = 0.53
    cont4 and cont8 = 0.53


Strong correlation is observed between the above pairs

This represents an opportunity to reduce the feature set through transformations such as PCA.

- Scatter plot

- Categorical attributes

## Data Preparation
- One Hot Encoding of categorical data


```python
import pandas

#cat1 to cat116 have strings. The ML algorithms we are going to study require numberical data
#One-hot encoding converts an attribute to a binary vector

#Variable to hold the list of variables for an attribute in the train and test data

cols = dataset.columns
labels = []

for i in range(0, split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))
```


```python
#Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:,i])
    feature = feature.reshape(dataset.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)
```

    (188318, 1176)



```python
dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)
print(dataset_encoded.shape)
```

    (188318, 1191)


- Split data into train and validation


```python
r,c = dataset_encoded.shape

i_cols = []

for i in range(0, c-1):
    i_cols.append(i)
    
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:, (c-1)]

val_size = 0.1

seed = 0

from sklearn import cross_validation

X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)

X_all = []

comb = []

mae = []

n = "All"

X_all.append([n, i_cols])
```

## Evaluation, prediction, and analysis

- Linear Regression


```python
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

model = LinearRegression(n_jobs=-1)
algo = "LR"

for name, i_cols_list in X_all:
    model.fit(X_train[:, i_cols_list], Y_train)
    result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
    mae.append(result)
    print(name + " %s" % result)
comb.append(algo)
```

    All 1277.35669997


- Ridge Regression


```python
from sklearn.linear_model import Ridge

a_list = numpy.array([1,2,3])

for alpha in a_list:
    model = Ridge(alpha=alpha,random_state=seed)
    
    algo = "Ridge"
    
    for name, i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
    comb.append(algo + " %s" % alpha)
```

    All 1267.53699309
    All 1265.12209912
    All 1263.67997896


- LASSO Linear Regression


```python
from sklearn.linear_model import Lasso

a_list = numpy.array([0.001, 0.003, 0.006])

for alpha in a_list:
    model = Lasso(alpha=alpha,random_state=seed)
    algo = "Lasso"
    
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
    comb.append(algo + " %s" % alpha)

```

    All 1262.53145951
    All 1272.56017154
    All 1299.67410024


- XGBoost


```python
from xgboost import XGBRegressor

n_list = numpy.array([100, 500, 1000])

for n_estimators in n_list:
    model = XGBRegressor(n_estimators=n_estimators,seed=seed)
    algo = "XGB"
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
    comb.append(algo + " %s" % n_estimators)
```

    All 1220.40564421
    All 1174.85933897
    All 1169.39060064



```python
fig, ax = plt.subplots()
plt.plot(mae)
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb, rotation='vertical')
plt.show()
```


![png](https://mllog.github.io/img/output_31_0.png)


## Make prediction with XGB


```python
dataset_test
```


```python
X = numpy.concatenate((X_train, X_val), axis=0)
del X_train
del X_val
Y = numpy.concatenate((Y_train, Y_val), axis=0)
del Y_train
del Y_val

n_estimators = 1000

best_model = XGBRegressor(n_estimators=n_estimators,seed=seed)
best_model.fit(X,Y)
del X
del Y
```


```python
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset_test.iloc[:,i])
    feature = feature.reshape(dataset_test.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

del cats

#Concatenate encoded attributes with continuous attributes
X_test = numpy.concatenate((encoded_cats,dataset_test.iloc[:,split:].values),axis=1)

#Make predictions using the best model
predictions = numpy.expm1(best_model.predict(X_test))
del X_test
# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))
```


```python

```
