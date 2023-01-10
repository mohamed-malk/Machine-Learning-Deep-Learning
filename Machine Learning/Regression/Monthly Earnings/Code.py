# code
#________________________

# load libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import curve_fit
from patsy import dmatrices

#Load Data from 'MonthlyEarnings.csv'
dataFrame = pd.read_csv('MonthlyEarnings.csv')

# Handling data  (Features Engineering)
def handle_data(dataFrame):
    ''' 
      

. @brief handling data
.
. remove every none-numeric value in data
.
. @param dataFrame --> input DataSet 
. 
. return  data after handling as pandas DataFrame 
.    
    '''
    cl = np.shape(dataFrame)[1] # number of columns
    for i in range(cl):dataFrame.iloc[:,i] = pd.to_numeric(dataFrame.iloc[:,i], errors='coerce')
    dataFrame = dataFrame.dropna()
    return dataFrame
dataFrame = handle_data(dataFrame)


## Preperation data for Linear Regression

features = dataFrame.iloc[:,:6]; target = dataFrame.iloc[:,-1]
#___________________________________________________________________________________________________________

# check on multicorrlinearity
def corrcoef(data):
    '''
       \f
   \
     corrcoef(data) 
. @brief apply the correlation coefficient matrix to a list of the datasets 
. 
.  work in datasets as list or pandas.core.frame.DataFrame , only 
.  it doesn't need to work transpose on dataset
.
.  1 denotes the correlation coefficient between the same dataset
.  The largest positive value in the matrix indicates a large relationship between the two groups
.  and it is a direct relationship, since if one of the features increases
.  the corresponding increase in it.  
.  The lowest negative value in the matrix indicates a large relationship between the two groups
.  which is an inverse relationship, whereby if one of the features increases
.  the corresponding decreases.
.  0 indicates that there is no relationship between the features expressed by the correlation coefficient between them  
.
. @param data is list of datasets or pandas.core.frame.DataFrame
. 
. return corrcoef matrix between datasets
.
    '''
    x = np.shape(data)[1] ## number of features (columns)
    corrcof_matrix = pd.DataFrame() ## corrcoef matrix
    corrcof_matrix['variables'] = data.columns
    for i in range(0,x):
        list = []
        for j in range(0,x):
            #### corrcoef equation
            matrix_1 =  ( ( data.iloc[:,i] - np.float(np.mean(data.iloc[:,i])) )   * ( data.iloc[:,j] - np.float(np.mean(data.iloc[:,j])) ) )
            matrix_2 = ( ( data.iloc[:,i] - np.float(np.mean(data.iloc[:,i]))  ) ** 2)
            matrix_3 = ( ( data.iloc[:,j] - np.float(np.mean(data.iloc[:,j]))  ) ** 2)
            s = np.sqrt( sum(matrix_2) * sum(matrix_3) )
            list.append(sum(matrix_1) / s);pass
        corrcof_matrix[data.columns[i]] = list;pass
    return corrcof_matrix

def VIF(features, target):
    strin = features.columns[0]
    for i in features.columns[1:]: strin += '+'+ i
    y, X = dmatrices(target.name + ' ~ ' + strin, data=pd.concat([features,target],axis=1), return_type='dataframe')
   
    vif = pd.DataFrame()
    vif['VIF'] = ["{:.1f}".format(variance_inflation_factor(X.values, i)) for i in range(X.shape[1])]
    vif['variable'] = X.columns
    return vif
#########################################

def objective(x,a,b): return a + b*x
def isHomoscedasticity(features, target):  
    res = [];cl = np.shape(features.columns)[0]
    for i in range(cl):
        var,max,min = [],[],[]; uni = np.unique(features[features.columns[i]])
        max = [target[ features[features.columns[i]] == j].max() for j in uni ]
        min = [target[ features[features.columns[i]] == j].min() for j in uni ]
        for j in range(len(min)):
            if max[j] != min[j]:var.append(max[j]-min[j])
        res.append(np.abs(np.max(var)/np.min(var)))
        py.scatter(features.iloc[:,i], target); 
        xLine = np.arange(features[features.columns[i]].min(), features[features.columns[i]].max(),1)
        p, _ = curve_fit(objective,features[features.columns[i]],target); a,b=p
        yLine = objective(xLine,a,b)
        for i in range(0,len(uni)):sns.lineplot(x=[uni[i],uni[i]],y=[min[i],max[i]]);
        py.plot(xLine,yLine,'--',color='red'); py.show()
        pass
    return res

# Build multi Regression model
def model(features,target):
    linear = LinearRegression()
    x_tr, x_t, y_tr, y_t = train_test_split(features,target,test_size=0.20,random_state=1)
    modl = linear.fit(x_tr, y_tr)
    y_pred = modl.predict(x_t)
    print(r2_score(y_t,y_pred))
    pass
##_____________________________________________________________________________________________________________




# Linear relation
#for i in range(x):
#    print(i,end='\n-----------------------------------\n')
#    print( np.corrcoef(features.iloc[:,i],target),end='\n###########################\n\n')

#features = pd.concat([features.iloc[:,0:3],features.iloc[:,3],features.iloc[:,5]],axis=1)

### check on autocorrelatin 
#x = np.count_nonzero(features.columns) 
#for i in range(x):print("{:.1f}".format(durbin_watson(features.iloc[:,i])), end =' -- ')
#print('\n----------------------------------\n')

#features = pd.concat([features.iloc[:,0:3],features.iloc[:,3],features.iloc[:,5]],axis=1)

### multi
#f = corrcoef(features)
#x = np.count_nonzero(f.columns)
#for i in range(x-1):
#    print(f.iloc[i,0], end='\n##########################################\n')
#    for j in range(1,x):
#        print("{:.4f}".format(f.iloc[i,j]), f.iloc[j-1,0],sep='\t');pass
#    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
#print('\n----------------------------------\n')
###   ###   # 
#print(VIF(features,target))
#print('\n----------------------------------\n')

##features = pd.concat([features.iloc[:,0],features.iloc[:,2],features.iloc[:,4]],axis=1)

### HOMO
##print(isHomoscedasticity(features, target))
##print('\n----------------------------------\n')


features = pd.concat([features.iloc[:,0],features.iloc[:,3]],axis=1)
model(features.iloc[:,0:2],target)







################################################

#     VIF   variable
#0  154.2  Intercept
#1    1.5  Knowledge
#2    1.2   YearsEdu
#3    1.2        Age

#  H  -- IQ  -- YE  -- YExp -- AGE 
# 0.1 -- 0.0 -- 0.1 -- 0.0 -- 0.0 --