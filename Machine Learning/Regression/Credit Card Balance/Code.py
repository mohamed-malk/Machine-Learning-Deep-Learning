# Dataset Link: https://www.kaggle.com/datasets/abdalrahmanelnashar/credit-card-balance-prediction
#_____________________________________________________________________________________________________

# load libraries
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import curve_fit
from patsy import dmatrices
#########################################################################################################

#Load Data from 'Task2_Poly_1.csv'
dataFrame = pd.read_csv('Task_4-files/CreditCardBalance.csv')

## convert into specific-numberic values
number = LabelEncoder()
dataFrame['Student'] = number.fit_transform(dataFrame['Student'].astype('str')) 

#dataFrame['Gender'] = number.fit_transform(dataFrame['Gender'].astype('str')) 
#dataFrame['Married'] = number.fit_transform(dataFrame['Married'].astype('str')) 
#dataFrame['Ethnicity'] = number.fit_transform(dataFrame['Ethnicity'].astype('str')) 
##################

# Rating - Cards - Education - Student
features = pd.concat([dataFrame.iloc[:,3:5],
     dataFrame.iloc[:,3]*dataFrame.iloc[:,4], # rat * card
     dataFrame.iloc[:,3]*dataFrame.iloc[:,6], # rat * edu
     dataFrame.iloc[:,6], dataFrame.iloc[:,8]], axis=1); target = dataFrame.iloc[:,-1]

#########################################################################################################

## split dataSet 
x_tr, x_t, y_tr, y_t = train_test_split(features, target, test_size=0.20, random_state=True)

#_ Linear Regression Model

linear = LinearRegression()
linear.fit(x_tr,y_tr)

y_linear_pre = linear.predict(x_t)
print(' Linear Regression Model ')
print('Coef : ', linear.coef_)
print('Intercept : ', linear.intercept_)
print('MSE : ', mean_squared_error(y_t,y_linear_pre))
print('R2-Score : ', r2_score(y_t,y_linear_pre))
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

#_ Lasso Regression Model
lasso = Lasso(alpha=1)
lasso.fit(x_tr,y_tr)

y_lasso_pre = lasso.predict(x_t)
print(' Lasso Regression Model ')
print('Coef : ', linear.coef_)
print('Intercept : ', linear.intercept_)
print('MSE : ', mean_squared_error(y_t,y_lasso_pre))
print('R2-Score : ', r2_score(y_t,y_lasso_pre))
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

#_ Ridge Regression Model
ridge = Ridge(alpha=1)
ridge.fit(x_tr,y_tr)

y_ridge_pre = ridge.predict(x_t)
print(' Ridge Regression Model ')
print('Coef : ', linear.coef_)
print('Intercept : ', linear.intercept_)
print('MSE : ', mean_squared_error(y_t,y_ridge_pre))
print('R2-Score : ', r2_score(y_t,y_ridge_pre))
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
############################################################################################
