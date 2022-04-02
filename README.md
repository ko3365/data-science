# Project 1 - Patient Classification
### Objective: Classify two classes 1. Patient with risk of heart disease 2. Patient without risk of heart disease
Main Difference between Classification and Linear Regression is that Classification is based on the probability which has range of [0,1] with (-inf,inf) domain.

Probability function:

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Gray}&space;P(Y=1)&space;=&space;\frac{1}{1-e^{-(\beta_0&plus;\beta_1X_{1}&plus;\hdots&plus;\beta_pX_p)}}&space;}">

Log-Odds:

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Gray}&space;log(\frac{P(Y=1)}{1-P(Y=1)})=&space;\beta_0&plus;\beta_1X_1&plus;\beta_2X_2&plus;\hdots&plus;\beta_pX_p}">

Data Description:
A subset of the UCI Heart Disease dataset is used to train the classification model. The dataset includes 13 Features and 1 Label:
- **age**: Age in years
- **sex**: 1=male;0=female
- **cp**: Chest pain type (0=asymptomatic;1=atypical angina;2=non-anginal pain;3=typical angina)
- **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital)
- **cholserum**: Cholestoral in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (0= showing probable or definite left ventricular hypertrophy by Estes' criteria; 1 = normal; 2 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV))
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeakST**: Depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment (0 = downsloping; 1 = flat; 2 = upsloping)
- **ca**: Number of major vessels (0-4) colored by flourosopy
- **thal**: 1 = normal; 2 = fixed defect; 3 = reversable defect
- **sick**: Indicates the presence of Heart disease (True = Disease; False = No disease)

## 1.1 Loading Data and Importing libraries
```Python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Training / Metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Drawing confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Helper function to draw confusion matrices
def draw_confusion_matrix(y, y_hat, title_name="Confusion Matrix"):
    '''Draws a confusion matrix for the given target and predictions'''
    cm = metrics.confusion_matrix(y, y_hat)
    metrics.ConfusionMatrixDisplay(cm).plot()
    plt.title(title_name)

df = pd.read_csv('heartdisease.csv')
```

## 1.2 Building Pipeline
Standardization for numerical features and hot-encoding the categorical features

```Python
categorical_features = ["cp","restecg","slope","ca","thal"]
numerical_features = ["age","trestbps","chol","thalach","oldpeak"]

num_pipeline = Pipeline([
    ('std_scaler',StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, numerical_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'),categorical_features)
])


features_transformed = full_pipeline.fit_transform(df)
```

### 1.3 Implementing Logistic Regression

```Python
log_reg = LogisticRegression()
log_reg.fit(x_train_tf, y_train)
log_reg_predicted_train = log_reg.predict(x_train_tf)
log_reg_predicted_test = log_reg.predict(x_test_tf)
```

# Project 2 - Sale Prediction

### Objective - Develop a predictive model to help forecast product sales
Data Description:
- BrandDetails.csv: details of which products are sold in each brand
- BrandTotalSales.csv: total monthly revenue of brands
- BrandTotalUnits.csv: total monthly units of products sold
- BrandAverageRetailPrice.csv: average retail price of brands
- Top50ProductsbyTotalSales-Timeseries.csv: list of top 50 products

### 2.1 Data Engineering
Since the rows of data are not independent (impact of previous month to current month), we need to condense the time series data into new features. Here, we added columns for previous month's sales, previous month's units, rolling averages of sales and units. 

Given the complexity of multiple brands with overlapping time intervals, the data is split by brands and then reassembled.

```Python
dataunits = pd.read_csv('data/BrandTotalUnits.csv')
#...#
brands = dataunits["Brands"].unique()
for brand in brands:
    if (i%100 == 0):
        print('{}th Brand Processing'.format(i))
    units = dataunits[dataunits.Brands == brand]

    units = units.assign(Previous_month_unit=units.loc[:,'Total Units'].shift(1))
    units = units.assign(Previous_Month2=units.loc[:,'Total Units'].shift(2))
    units = units.assign(Previous_Month3=units.loc[:,'Total Units'].shift(3))
    units = units.fillna(0)
    units = units.assign(Rolling_avg_unit=(units.Previous_month_unit+units.Previous_Month2+units.Previous_Month3)/3)
    units = units.drop(columns=['Previous_Month2','Previous_Month3'])
    #...#
```
Hot-encoding the brands (categorical variable) will create 1640 additional columns. Instead, the brands are ranked by their average market share. Rank has ordinal property (has natural ordering) so it can be directly used to training the linear regression model.

```Python
market_share_avg_set = [0 if math.isnan(x) else x for x in market_share_avg_set]
rank_list = rankdata(market_share_avg_set,method = 'min')
j = 0
for brand in brands:
    if(j%200==0):
        print('{}th Brand Processing'.format(j))
    final_data.loc[final_data['Brands']==brand,'Rank'] = rank_list[j]
    j=j+1
print('Ranking Completed')
```

### 2.2 Train & Test Data Split by Brands
To test the performance of the predictive model, train and test data are separated by brands so that the training data does not contain the same brands that are in the test data. This will show how well the model predicts the sales with the data that it has never seen before

```Python
def train_test_split_by_brands(xi,yi,brands_list,train_i,test_i):
    
    x_train = xi.loc[xi['Brands'].isin(brands_list[train_i])]
    x_test = xi.loc[xi['Brands'].isin(brands_list[test_i])]
    y_train_pd = yi.loc[yi['Brands'].isin(brands_list[train_i])]
    y_test_pd = yi.loc[yi['Brands'].isin(brands_list[test_i])]

    x_train = x_train.drop(columns='Brands')
    x_test = x_test.drop(columns='Brands')
    return x_train, x_test, y_train_pd['Total Sales ($)'].tolist(),y_test_pd['Total Sales ($)'].tolist()
```

### 2.3. Model Fitting 
Linear Regression method is used to fit the model

```Python
x_train_piped_sm =sm.add_constant(x_train_piped)
ols = sm.OLS(y_train,x_train_piped_sm)
result = ols.fit()
x_test_piped_sm =sm.add_constant(x_test_piped)
ypred = result.predict(x_test_piped_sm)
```
