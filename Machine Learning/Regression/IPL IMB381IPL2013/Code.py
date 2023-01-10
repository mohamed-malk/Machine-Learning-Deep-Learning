

# load libraries
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as py
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import  LabelEncoder , StandardScaler
from dtreeviz.trees import *
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
####################################################


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
    #strin = features.columns[0]
    #for i in features.columns[1:]: strin += '+'+ i
    #y, X = dmatrices('SOLD PRICE' + ' ~ ' + strin, data=pd.concat([features,target],axis=1), return_type='dataframe') 
    vif = pd.DataFrame(); X = features
    vif['VIF'] = ["{:.1f}".format(variance_inflation_factor(X.values, i)) for i in range(X.shape[1])]
    vif['variable'] = X.columns
    vif = vif.sort_values(by='VIF');
    return vif

##########################################

#####################################################
# load data
dataFrame = pd.read_csv('Task_6-files/IPL IMB381IPL2013.csv')
x = dataFrame.iloc[:,:-1]; y = dataFrame.iloc[:,-1] 


# Data exploration
number = LabelEncoder()

#_ AGE

# [1, 2, 3]
# [16, 86, 28]
# [12.30, 66.15, 21.53]%
# 1800000  -  24000  ###  720,250.0
# 1800000  -  20000  ###  484,534.88372093026
# 1800000  -  100000 ###  520,178.5714285714
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#_ PLAYING ROLE - CAPTAINCY EXP - AUCTION YEAR

#name_=np.unique(x['PLAYING ROLE'])
#print(name_)
#['Allrounder' 'Batsman' 'Bowler' 'W. Keeper']
#[    35          39        44        12     ]
# 1,550,000 - 20,000  ### 519,571.4285714286
# 1,800,000 - 50,000  ### 647,435.8974358974 $
# 950,000 - 24,000    ### 419,977.2727272727
# 1,500,000 - 100,000 ### 487,083.3333333333
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#name_= np.unique(x['CAPTAINCY EXP'])
# [ 0  1  ]
# [ 89 41 ]
# [ 68.46 31.53 ]
# 1,600,000 - 20,000 ### 433,528.08988764044
# 1,800,000 - 50,000 ### 711,585.3658536585
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#name_= np.unique(x['AUCTION YEAR'])
# [ 2008 2009 2010 2011 ]
# [  75   10    3   42  ]
# [ 57.69 7.69 2.30 32.30 ]%
# 1,500,000 - 50,000 ### 492,066.6666666667
# 1,550,000 - 24,000 ### 458,400.0
# 750,000  - 20,000 ### 290,000.0
# 1800,000 - 50,000 ### 604,761.9047619047 $

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#_ COUNTRY - TEAM

#name_= np.unique(x['COUNTRY'])
# ['AUS' 'BAN' 'ENG' 'IND' 'NZ' 'PAK' 'SA' 'SL' 'WI' 'ZIM']
# [ 22     1     3     53    7    9    16   12    6    1  ]
# [ 16.9  0.7   2.3   40.7   5.3  6.9  12.3  9.2  4.6  0.7 ]%
# 1350000  -  20000  ###  434,090.9090909091
# 50000  -  50000  ###  50,000.0
# 1,550,000  -  100,000  ###  1,066,666.6666666667 $$$$$$$
# 1800000  -  24000  ###  652,339.6226415094
# 1000000  -  160000  ### 526,428.5714285715
# 675000  -  100000  ###  330,555.55555555556
# 950000  -  50000  ###   427,500.0
# 975000  -  100000  ###  423,750.0
# 800000  -  100000  ###  279,166.6666666667
# 125000  -  125000  ###  125,000.0
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#name_= np.unique(x['TEAM']);print(name_)
# ['CSK' 'CSK+' 'DC' 'DC+' 'DD' 'DD+' 'KKR' 'KKR+' 'KXI+' 'KXIP' 'KXIP+' 'MI' 'MI+' 'RCB' 'RCB+' 'RR' 'RR+']
# [ 14     5     7    10    6    10     5   12       1       5     7      6     6     9     12     6   9   ]
#1550000  -  50000  ###  601,071.4285714285
#675000  -  290000  ###  478,000.0
#750000  -  100000  ###  365,714.28571428574
#1350000  -  100000  ###  585,000.0
#1800000  -  225000  ###  712,500.0
#850000  -  80000  ###  498,000.0
#425000  -  125000  ###  290,000.0
#950000  -  50000  ###  495,000.0
#900000  -  900000  ###  900,000.0  $$$
#400000  -  50000  ###  245,000.0
#1800000  -  50000  ###  677,857.1428571428
#1800000  -  150000  ###  779,166.6666666666
#1600000  -  200000  ###  608,333.3333333334
#1800000  -  20000  ###  442,222.22222222225
#1550000  -  50000  ###  558,750.0
#950000  -  100000  ###  420,833.3333333333
#950000  -  24000  ###  388,222.22222222225
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

py.scatter(x['BASE PRICE'], y)
py.show() #__ 15.6843
######print(xx['BASE PRICE'][x['BASE PRICE'] > 400000].count()) #@ 3 rows
dataFrame = dataFrame[x['BASE PRICE'] <= 400000]
x = dataFrame.iloc[:,:-1]; y = dataFrame.iloc[:,-1] 
#py.scatter(x['BASE PRICE'], y)
#py.show() #__ 78.709
#corr with y = 0.523
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

x_num = x.select_dtypes(include='number')
x_num = x_num.drop(['Sl.NO.', 'AGE', 'AUCTION YEAR', 'BASE PRICE', 'CAPTAINCY EXP'], axis=1);

# ODI-SR-B : 0.2  ^^ AVE : 0.4 ^^ SR-B : 0.2
x_num = x_num.drop(['ODI-SR-B', 'AVE', 'SR-B'], axis=1);

##____ after VIF and cor

#f = corrcoef(x)
#x_ = np.count_nonzero(f.columns)
#for i in range(x_-1):
#    print(f.iloc[i,0], end='\n##########################################\n')
#    for j in range(1,x):
#        print("{:.4f}".format(f.iloc[i,j]), f.iloc[j-1,0],sep='\t');pass
#    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
#print('\n----------------------------------\n')
#print(VIF(x,y))

x_num = x_num.drop(['HS', 'RUNS-S', 'ODI-RUNS-S','ODI-WKTS','RUNS-C','AVE-BL','SR-BL'], axis=1); 

#___________________________________#####____________________________________#

x['COUNTRY'] = number.fit_transform(x['COUNTRY'])
x['TEAM'] = number.fit_transform(x['TEAM'])
x['PLAYING ROLE'] = number.fit_transform(x['PLAYING ROLE'])

x = pd.concat([x['COUNTRY'], x['TEAM'], x['PLAYING ROLE'], x['CAPTAINCY EXP'], x['AUCTION YEAR'], x['BASE PRICE'],x_num], axis =1)

# before >>> 78.3006
# OLS (p-value) ::
# ODI-SR-BL = 0.942 < after remove >> 78.3416
# T-WKTS = 0.868 < after remove >> 78.709
# ECON = 0.827 < after remove >> 78.38
x = x.drop(['ODI-SR-BL', 'T-WKTS'], axis=1); 


x_tr, x_t, y_tr, y_t = train_test_split(x, y,random_state=1, test_size=0.20)

ml = sm.OLS(y_tr,x_tr).fit(); print(ml.summary())

model_ = LinearRegression()
model_.fit(x_tr, y_tr)
pre = model_.predict(x_t)
print(r2_score(y_t, pre)*100)