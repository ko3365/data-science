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
