# Datset Link: https://www.kaggle.com/datasets/brendan45774/test-file
#_____________________

# load libraries
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import curve_fit
from patsy import dmatrices

#Load Data from 'Task2_Poly_1.csv'
dataFrame = pd.read_csv('Survived_Train.csv'); data_test = pd.read_csv('Survived_Test.csv'); target_test_output = pd.read_csv('Survived_TestActualOutput.csv').iloc[:,1]
#####################################################################
#Index(['Sex', 'Embarked'], dtype='object')
####> Handling data  (Features Engineering)

## Age <> feature
dataFrame['Age'] = pd.to_numeric(dataFrame['Age'], errors='coerce')
dataFrame['Age'] = dataFrame['Age'].replace(np.NAN,np.mean(dataFrame['Age']))
#
data_test['Age'] = pd.to_numeric(data_test['Age'], errors='coerce')
data_test['Age'] = data_test['Age'].replace(np.NAN,np.mean(data_test['Age']))

## SibSp <> feature
dataFrame['SibSp'] = pd.to_numeric(dataFrame['SibSp'], errors='coerce')
dataFrame['SibSp'] = dataFrame['SibSp'].replace(np.NAN,np.mean(dataFrame['SibSp']))
#
data_test['SibSp'] = pd.to_numeric(data_test['SibSp'], errors='coerce')
data_test['SibSp'] = data_test['SibSp'].replace(np.NAN,np.mean(data_test['SibSp']))

### Parch <> feature
#dataFrame['Parch'] = pd.to_numeric(dataFrame['Parch'], errors='coerce')
#dataFrame['Parch'] = dataFrame['Parch'].replace(np.NAN,np.mean(dataFrame['Parch']))
##
#data_test['Parch'] = pd.to_numeric(data_test['Parch'], errors='coerce')
#data_test['Parch'] = data_test['Parch'].replace(np.NAN,np.mean(data_test['Parch']))

### Fare <> feature
#dataFrame['Fare'] = pd.to_numeric(dataFrame['Fare'], errors='coerce')
#dataFrame['Fare'] = dataFrame['Fare'].replace(np.NAN,np.mean(dataFrame['Fare']))
##
#data_test['Fare'] = pd.to_numeric(data_test['Fare'], errors='coerce')
#data_test['Fare'] = data_test['Fare'].replace(np.NAN,np.mean(data_test['Fare']))


## convert into specific-numberic values
number = LabelEncoder()

#dataFrame['PassengerId'] = number.fit_transform(dataFrame['PassengerId'].astype('int'))+1
#dataFrame['Pclass'] = number.fit_transform(dataFrame['Pclass'].astype('int'))+1
dataFrame['Sex'] = number.fit_transform(dataFrame['Sex'].astype('str')) # 1 for 'male' || 0 for 'female' 
#dataFrame['Embarked'] = number.fit_transform(dataFrame['Embarked'].astype('str')) # 2 for 'S' ||1 for 'Q' || 0 for 'C' 
#
#data_test['PassengerId'] = number.fit_transform(data_test['PassengerId'].astype('int'))+1
#data_test['Pclass'] = number.fit_transform(data_test['Pclass'].astype('int'))+1
data_test['Sex'] = number.fit_transform(data_test['Sex'].astype('str')) # 1 for 'male' || 0 for 'female' 
#data_test['Embarked'] = number.fit_transform(data_test['Embarked'].astype('str')) # 2 for 'S' ||1 for 'Q' || 0 for 'C' 


########_____________________________________________________________________________________________________________

features, target = pd.concat([dataFrame.iloc[:,2],dataFrame.iloc[:,4:8]],axis=1), dataFrame.iloc[:,1]
features_test = pd.concat([data_test.iloc[:,1],data_test.iloc[:,3:7]],axis=1)

features = pd.concat([# 1 2 3 
    features.iloc[:,1:4],
    features.iloc[:,1]*features.iloc[:,2],
    features.iloc[:,2]*features.iloc[:,3],

    ],
                     axis=1)
features_test = pd.concat([
    features_test.iloc[:,1:4],
    features_test.iloc[:,1]*features_test.iloc[:,2],
    features_test.iloc[:,2]*features_test.iloc[:,3],
    ],
                     axis=1)

print(features.columns)
#[ Sex, Age, SibSp ]

# Build Logistic Regression model
logistic = LogisticRegression()
modl = logistic.fit(features, target)
y_pred = modl.predict(features_test)
#print("R2-Score = ", r2_score(target_test_output,y_pred) * 100)
#print("accuracy_score = ", accuracy_score(target_test_output,y_pred) * 100)
#print("confusion_matrix :\n", confusion_matrix(target_test_output,y_pred))
#print("classification_report =:\n", classification_report(target_test_output,y_pred))

sex = int(input("Enter 0 : female ----- 1 : male >> "))
age = float(input("Enter Age >> "))
sibSp = int(input("Enter SibSp >> "))
pre = modl.predict([[sex,age,sibSp,sex*age,age*sibSp]])

if pre == 0 :print('died')
else :print('Survived')



