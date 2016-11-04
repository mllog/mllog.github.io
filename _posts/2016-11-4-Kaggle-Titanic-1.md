---
layout: post
title: Start ML Journey with Titanic
subtitle: A Notebook for Kaggle Titanic Competition
tags: [machine learning, kaggle]
---


Tweak the style of the notebook to have centered plots


```python
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
```





<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>




# I - Exploratory data analysis

Import some useful libraries


```python
# remove warnings
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
pd.options.display.max_rows = 100
```

Loading the data set


```python
data = pd.read_csv('data/train.csv')
```


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Pandas allows us to statistically describe numerical features using the describe method.


```python
data.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



Age column has missing values. A solution is to replace the null values with the median age


```python
data['Age'].fillna(data['Age'].median(), inplace=True)
```


```python
data.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.361582</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>13.019697</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>35.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



Draw some charts to understand more about the data


```python
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex, dead_sex])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(15, 8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f54ff914898>




![png](output_14_1.png)


The Sex variable seems to be a decisive feature. Women are more likely to survive

Let's now correlate the suvival with the age variable


```python
figure = plt.figure(figsize=(15, 8))
plt.hist([data[data['Survived']==1]['Age'],
          data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f54ff8b5ac8>




![png](output_16_1.png)


Let's now focus on the Fare ticket


```python
figure = plt.figure(figsize=(15, 8))
plt.hist([data[data['Survived']==1]['Fare'], 
          data[data['Survived']==0]['Fare']], stacked=True, 
         color=['g', 'r'], bins=30, label=['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f54ff8b5a58>




![png](output_18_1.png)


Combine the age, the fare and the survival on a single chart


```python
plt.figure(figsize=(15, 8))
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'], 
           data[data['Survived']==1]['Fare'], c='green', s=40)
ax.scatter(data[data['Survived']==0]['Age'], 
           data[data['Survived']==0]['Fare'], c='red', s= 40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived', 'dead'), scatterpoints=1, loc='upper righ', fontsize=15)
```




    <matplotlib.legend.Legend at 0x7f54fd247438>




![png](output_20_1.png)


The fare is correlated with the Pclass


```python
ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15, 8), ax = ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f54fd2e84a8>




![png](output_22_1.png)


Let's now see how the embarkation site affects the survival


```python
survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark, dead_embark])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(15,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f54fd44c2b0>




![png](output_24_1.png)


There seems to be no distinct correlation here

# II - Feature Enginerring

Function that asserts whether or not a feature has been processed.


```python
def status(feature):
    print('Processing %s :ok' %(feature))
```

## Loading data
Load and combine train set and test set. Combined set will be tranning set for a model


```python
def get_combined_data():
    # reading train data
    train = pd.read_csv('data/train.csv')
    
    # reading test data
    test = pd.read_csv('data/test.csv')
    
    # extracting and then removing the targets from the traing data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)
    
    # Merging train data and test data for future engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    return combined
```


```python
combined = get_combined_data()
```


```python
combined.shape
```




    (1309, 11)




```python
combined.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Extracting the passenger titles


```python
def get_titles():
    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
```


```python
get_titles()
```


```python
combined.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>



## Processing the ages

The are 177 values missing for Age. We need to fill the missing value


```python
grouped = combined.groupby(['Sex', 'Pclass', 'Title'])
grouped.median()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>PassengerId</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th>Pclass</th>
      <th>Title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">female</th>
      <th rowspan="4" valign="top">1</th>
      <th>Miss</th>
      <td>529.5</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>99.9625</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>853.5</td>
      <td>45.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>78.1125</td>
    </tr>
    <tr>
      <th>Officer</th>
      <td>797.0</td>
      <td>49.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.9292</td>
    </tr>
    <tr>
      <th>Royalty</th>
      <td>760.0</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>86.5000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>Miss</th>
      <td>606.5</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.2500</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>533.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>26.0000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>Miss</th>
      <td>603.5</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>668.5</td>
      <td>31.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15.5000</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">male</th>
      <th rowspan="4" valign="top">1</th>
      <th>Master</th>
      <td>803.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>134.5000</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>634.0</td>
      <td>41.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>47.1000</td>
    </tr>
    <tr>
      <th>Officer</th>
      <td>678.0</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.5500</td>
    </tr>
    <tr>
      <th>Royalty</th>
      <td>600.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27.7208</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>Master</th>
      <td>550.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.0000</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>723.5</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>Officer</th>
      <td>513.0</td>
      <td>41.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>Master</th>
      <td>789.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>22.3583</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>640.5</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.8958</td>
    </tr>
  </tbody>
</table>
</div>



Look at the median age column and see how this value can be different based on the Sex, Pclass and Title put together.

For example:
- If the passenger is female, from Pclass 1, and from royalty the median age is 39.
- If the passenger is male, from Pclass 3, with a Mr title, the median age is 26.

Let's create a function that fills in the missing age in **combined** based on these different attributes.


```python
def process_age():
    global combined
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')
```


```python
process_age()
```

    Processing age :ok



```python
combined.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Name           1309 non-null object
    Sex            1309 non-null object
    Age            1309 non-null float64
    SibSp          1309 non-null int64
    Parch          1309 non-null int64
    Ticket         1309 non-null object
    Fare           1308 non-null float64
    Cabin          295 non-null object
    Embarked       1307 non-null object
    Title          1309 non-null object
    dtypes: float64(2), int64(4), object(6)
    memory usage: 122.8+ KB


There are still some missing value in Fare, Embarked, Cabin


```python
def process_names():
    global combined
    
    combined.drop('Name', inplace=True, axis=1)
    title_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, title_dummies], axis=1)
    
    combined.drop('Title', axis=1, inplace=True)
    
    status('names')
```


```python
process_names()
```

    Processing names :ok



```python
combined.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title_Master</th>
      <th>Title_Miss</th>
      <th>Title_Mr</th>
      <th>Title_Mrs</th>
      <th>Title_Officer</th>
      <th>Title_Royalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Processing Fare


```python
def process_fares():
    
    global combined
    
    combined.Fare.fillna(combined.Fare.mean(), inplace=True)
    
    status('fare')
```


```python
process_fares()
```

    Processing fare :ok


## Processing Embarked
Fill missing values with the most frequent Embarked value.


```python
def process_embarked():
    global combined
    
    combined.Embarked.fillna('S', inplace=True)
    
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    
    status('embarked')
```


```python
process_embarked()
```

    Processing embarked :ok


## Processing Cabin
Fill missing cabins with U (for Unknown)


```python
def process_cabin():
    global combined
    
    combined.Cabin.fillna('U',inplace=True)
    
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    
    combined = pd.concat([combined, cabin_dummies], axis=1)
    
    combined.drop('Cabin', axis=1, inplace=True)
    
    status('cabin')
```


```python
process_cabin()
```

    Processing cabin :ok



```python
combined.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 26 columns):
    PassengerId      1309 non-null int64
    Pclass           1309 non-null int64
    Sex              1309 non-null object
    Age              1309 non-null float64
    SibSp            1309 non-null int64
    Parch            1309 non-null int64
    Ticket           1309 non-null object
    Fare             1309 non-null float64
    Title_Master     1309 non-null float64
    Title_Miss       1309 non-null float64
    Title_Mr         1309 non-null float64
    Title_Mrs        1309 non-null float64
    Title_Officer    1309 non-null float64
    Title_Royalty    1309 non-null float64
    Embarked_C       1309 non-null float64
    Embarked_Q       1309 non-null float64
    Embarked_S       1309 non-null float64
    Cabin_A          1309 non-null float64
    Cabin_B          1309 non-null float64
    Cabin_C          1309 non-null float64
    Cabin_D          1309 non-null float64
    Cabin_E          1309 non-null float64
    Cabin_F          1309 non-null float64
    Cabin_G          1309 non-null float64
    Cabin_T          1309 non-null float64
    Cabin_U          1309 non-null float64
    dtypes: float64(20), int64(4), object(2)
    memory usage: 266.0+ KB



```python
combined.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Title_Master</th>
      <th>Title_Miss</th>
      <th>Title_Mr</th>
      <th>Title_Mrs</th>
      <th>Title_Officer</th>
      <th>Title_Royalty</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Cabin_A</th>
      <th>Cabin_B</th>
      <th>Cabin_C</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Processing Sex 


```python
def process_sex():
    
    global combined
    
    combined['Sex'] = combined['Sex'].map({'male':1, 'female': 0})
    
    status('sex')
```


```python
process_sex()
```

    Processing sex :ok


## Processing Pclass


```python
def process_pclass():
    
    global combined
    
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
    combined = pd.concat([combined, pclass_dummies], axis=1)
    
    combined.drop('Pclass', axis=1, inplace=True)
    
    status('pclass')
```


```python
process_pclass()
```

    Processing pclass :ok


## Processing Ticket


```python
def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = list(map(lambda t : t.strip() , ticket))
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')
```


```python
process_ticket()
```

    Processing ticket :ok


## Processing Family


```python
def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    
    status('family')
```


```python
process_family()
```

    Processing family :ok



```python
combined.shape
```




    (1309, 68)




```python
combined.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Title_Master</th>
      <th>Title_Miss</th>
      <th>Title_Mr</th>
      <th>Title_Mrs</th>
      <th>Title_Officer</th>
      <th>Title_Royalty</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Cabin_A</th>
      <th>Cabin_B</th>
      <th>Cabin_C</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Ticket_A</th>
      <th>Ticket_A4</th>
      <th>Ticket_A5</th>
      <th>Ticket_AQ3</th>
      <th>Ticket_AQ4</th>
      <th>Ticket_AS</th>
      <th>Ticket_C</th>
      <th>Ticket_CA</th>
      <th>Ticket_CASOTON</th>
      <th>Ticket_FC</th>
      <th>Ticket_FCC</th>
      <th>Ticket_Fa</th>
      <th>Ticket_LINE</th>
      <th>Ticket_LP</th>
      <th>Ticket_PC</th>
      <th>Ticket_PP</th>
      <th>Ticket_PPP</th>
      <th>Ticket_SC</th>
      <th>Ticket_SCA3</th>
      <th>Ticket_SCA4</th>
      <th>Ticket_SCAH</th>
      <th>Ticket_SCOW</th>
      <th>Ticket_SCPARIS</th>
      <th>Ticket_SCParis</th>
      <th>Ticket_SOC</th>
      <th>Ticket_SOP</th>
      <th>Ticket_SOPP</th>
      <th>Ticket_SOTONO2</th>
      <th>Ticket_SOTONOQ</th>
      <th>Ticket_SP</th>
      <th>Ticket_STONO</th>
      <th>Ticket_STONO2</th>
      <th>Ticket_STONOQ</th>
      <th>Ticket_SWPP</th>
      <th>Ticket_WC</th>
      <th>Ticket_WEP</th>
      <th>Ticket_XXX</th>
      <th>FamilySize</th>
      <th>Singleton</th>
      <th>SmallFamily</th>
      <th>LargeFamily</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Scale features


```python
def scale_all_features():
    
    global combined
    
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print('Features scaled sucessfully!')
```


```python
scale_all_features()
```

    Features scaled sucessfully!


# III - Modeling

We'll be using Random Forests. Random Froests has proven a great efficiency in Kaggle competitions.

For more details about why ensemble methods perform well, you can refer to these posts:
- http://mlwave.com/kaggle-ensembling-guide/
- http://www.overkillanalytics.net/more-is-always-better-the-power-of-simple-ensembles/

Steps:
1. Break the combined dataset to train set and test set
2. Use the train set to build a predictive model
3. Evaluate the model using the train set
4. Test the model using the test set


```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
```

We use 5-fold cross validation with the Accuracy metric


```python
def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
```


```python
def recover_train_test_target():
    train0 = pd.read_csv('data/train.csv')
    
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    return train,test,targets
```


```python
train,test,targets = recover_train_test_target()
```

## Feature selection

We select features from 68 features:

- This decreases redundancy among the data
- This speeds up the training process
- This reduces overfitting

Tree-based estimators can be used to compute feature importances


```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)
```


```python
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
```


```python
features.sort(['importance'], ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PassengerId</td>
      <td>0.128622</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Age</td>
      <td>0.120738</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fare</td>
      <td>0.113315</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>0.111784</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Title_Mr</td>
      <td>0.105217</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Title_Miss</td>
      <td>0.038876</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Title_Mrs</td>
      <td>0.037361</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Pclass_3</td>
      <td>0.036786</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Cabin_U</td>
      <td>0.028874</td>
    </tr>
    <tr>
      <th>66</th>
      <td>SmallFamily</td>
      <td>0.022793</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Pclass_1</td>
      <td>0.020920</td>
    </tr>
    <tr>
      <th>64</th>
      <td>FamilySize</td>
      <td>0.019720</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SibSp</td>
      <td>0.018080</td>
    </tr>
    <tr>
      <th>67</th>
      <td>LargeFamily</td>
      <td>0.017819</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Parch</td>
      <td>0.015108</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Pclass_2</td>
      <td>0.013490</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Title_Master</td>
      <td>0.013395</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Embarked_S</td>
      <td>0.012483</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Ticket_XXX</td>
      <td>0.011821</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Embarked_C</td>
      <td>0.010695</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Cabin_E</td>
      <td>0.009665</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Singleton</td>
      <td>0.009049</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cabin_B</td>
      <td>0.007602</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Title_Officer</td>
      <td>0.007445</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Embarked_Q</td>
      <td>0.006894</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Cabin_C</td>
      <td>0.006703</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Ticket_PC</td>
      <td>0.006438</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Ticket_SWPP</td>
      <td>0.006282</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Cabin_D</td>
      <td>0.006055</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Ticket_STONO</td>
      <td>0.005963</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Ticket_A5</td>
      <td>0.003407</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Ticket_CA</td>
      <td>0.003235</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Ticket_WC</td>
      <td>0.002706</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Cabin_A</td>
      <td>0.002313</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Ticket_C</td>
      <td>0.001923</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Cabin_F</td>
      <td>0.001900</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Ticket_STONO2</td>
      <td>0.001834</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Ticket_SOPP</td>
      <td>0.001802</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Ticket_SOTONOQ</td>
      <td>0.001746</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Cabin_G</td>
      <td>0.001318</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Title_Royalty</td>
      <td>0.001016</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Ticket_WEP</td>
      <td>0.000827</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Ticket_PP</td>
      <td>0.000763</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Ticket_FC</td>
      <td>0.000733</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Ticket_LINE</td>
      <td>0.000677</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Ticket_SCPARIS</td>
      <td>0.000621</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Ticket_SOC</td>
      <td>0.000539</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ticket_A4</td>
      <td>0.000498</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Ticket_FCC</td>
      <td>0.000476</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Ticket_SCParis</td>
      <td>0.000330</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Cabin_T</td>
      <td>0.000321</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Ticket_SCAH</td>
      <td>0.000244</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Ticket_SOP</td>
      <td>0.000174</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Ticket_SOTONO2</td>
      <td>0.000101</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Ticket_SC</td>
      <td>0.000099</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Ticket_SP</td>
      <td>0.000092</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Ticket_PPP</td>
      <td>0.000077</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Ticket_SCOW</td>
      <td>0.000068</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Ticket_AS</td>
      <td>0.000052</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Ticket_Fa</td>
      <td>0.000047</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Ticket_SCA4</td>
      <td>0.000045</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ticket_CASOTON</td>
      <td>0.000017</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Ticket_SCA3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Ticket_STONOQ</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Ticket_LP</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Ticket_AQ4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Ticket_AQ3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Ticket_A</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now we transform the train set and test set in a more compact datasets.


```python
model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
train_new.shape
```




    (891, 15)




```python
test_new = model.transform(test)
test_new.shape
```




    (418, 15)



## Hyperparameters tuning

As mentioned in the beginning of the Modeling part, we will be using a Random Forest model.

Random Forest are quite handy. They do however come with some parameters to tweak in order to get an optimal model for the prediction task.

To learn more about Random Forests, you can refer to this link: https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/


```python
forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {'max_depth' : [4,5,6,7,8],
                  'n_estimators':[200,210,240,250],
                  'criterion':['gini', 'entropy']}
cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
```

    Best score: 0.8338945005611672
    Best parameters: {'criterion': 'gini', 'max_depth': 4, 'n_estimators': 200}


Now we generate solution for sumission


```python
output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('data/output_10.csv',index=False)
```


```python

```
