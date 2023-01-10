# load libraries
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as py
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import  LabelEncoder , StandardScaler
from dtreeviz.trees import *
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
####################################################

# load data
dataFrame = pd.read_csv('Task_6-files/CAR DETAILS FROM CAR DEKHO.csv')
x = pd.concat([dataFrame.iloc[:,:2],dataFrame.iloc[:,3:]],axis=1); y = dataFrame.iloc[:,2] 


# Data exploration
number = LabelEncoder()
std = StandardScaler()
#_ name
name = x['name']
name = name.str.split(" ", expand = True)
make = name[0]; make = pd.Series(number.fit_transform(make), name='make')
model = name[1]; model = pd.Series(number.fit_transform(model), name='model')


#_ year
x['year'] = std.fit_transform(np.array(x['year']).reshape(-1,1))


#_ km_driven
km = x['km_driven']
ma = max(km); mi = min(km)
vm = (ma + mi) / 2
va = (ma + vm) / 2; vi = (mi + vm) / 2

#_ convert to classes
s = len(km); km_ = []
for i in km:
    if   i <= ma and i > va: km_.append(0)  # [201650.5 1] is 0
    elif i <= va and i > vm: km_.append(1)  # [403300.0 201650.5] is 1
    elif i <= vm and i > vi: km_.append(2)  # [604949.5 403300.0] is 2
    elif i <= vi and i >= mi: km_.append(3) # [806599 604949.5] is 3
    pass
km_ = pd.Series(km_,name='km_driven')


#_ fuel         #_ transmission
f = x['fuel']; tr = x['transmission']

#_ make  << fuel_transmission column >>
f = pd.Series(number.fit_transform(f)); tr = pd.Series(number.fit_transform(tr))
f_tr = []; s = len(tr)
for i in range(s):
    if   f[i] == 0: f_tr.append(0) # CNG (only manual) is 0
    elif f[i] == 3: f_tr.append(1) # LPG (only manual) is 1
    elif f[i] == 2: f_tr.append(2) # Electric (only Automatic) is 2

    elif f[i] == 1 and tr[i] == 0: f_tr.append(3) # Diesel & Automatic is 3
    elif f[i] == 1 and tr[i] == 1: f_tr.append(4) # Diesel & Manual is 4
    
    elif f[i] == 4 and tr[i] == 0: f_tr.append(5) # Petrol & Automatic is 5
    elif f[i] == 4 and tr[i] == 1: f_tr.append(6) # Petrol & Manual is 6
    pass
f_tr = pd.Series(f_tr,name= 'fuel_transmission')

#_ seller_type
st = x['seller_type'];  st = pd.Series(number.fit_transform(st), name='seller_type')

#_ owner
o = x['owner']; o = pd.Series(number.fit_transform(o), name='owner')




#########################################################################################################

x = pd.concat([x['year'], km_, f_tr, st, o, model, make], axis=1)

#######################################################################################
x_tr, x_t, y_tr, y_t = train_test_split(x, y,random_state=1, test_size=0.20)

modl=sm.OLS(y_tr, x_tr).fit(); print(modl.summary())

randomforest = RandomForestRegressor(random_state=1)
model_ = randomforest.fit(x_tr, y_tr)
pre = model_.predict(x_t)
print(r2_score(y_t, pre)*100)
##########################################################################################







######___________________________ km_driven
#kmva = km.loc[(km <= ma) & (km > va)]
#ymva = y.loc[(km <= ma) & (km > va)]
##
#kvam = km.loc[(km <= va) & (km > vm)]
#yvam = y.loc[(km <= va) & (km > vm)]
##
#kvmi = km.loc[(km <= vm) & (km > vi)]
#yvmi = y.loc[(km <= vm) & (km > vi)]
##
#kvim = km.loc[(km <= vi) & (km >= mi)]
#yvim = y.loc[(km <= vi) & (km >= mi)]
####____________________________________________________________









######___________________________ fuel
# ['CNG' 'Diesel' 'Electric' 'LPG' 'Petrol']
# [ 40     2153       1        23    2123  ]
# [ 0.92   49.60      0.02     0.52   48.91  ]%
#_
# 277,174.925 C
# 669,094.2522062239 $
# 310,000.0 E
# 167,826.04347826086 L
# 344,840.1375412153
####____________________________________________________________








######___________________________ transmission
# ['Automatic' 'Manual']
# [   448        3892  ]
# [ 1,408,154.0    400,066.6857656732 ]
####____________________________________________________________







######___________________________ seller_type
# ['Dealer' 'Individual' 'Trustmark Dealer']
# [   994       3244              102      ]
# [   23.17     75.61             2.37     ]%
# Dealer             $ 721,822.8903420523
# Individual         $ 424,505.41923551174
# Trustmark Dealer   $ 914,950.9803921569
####____________________________________________________________







######___________________________ owner
# ['First Owner' 'Fourth & Above Owner' 'Second Owner' 'Test Drive Car' 'Third Owner']
# [    2832              81                  1106              17            304     ]
# [    65.25            1.86                 25.48            0.39           7.00    ]%
# First Owner             $ 598,636.9696327683
# Fourth & Above Owner    $ 173,901.19753086418
# Second Owner            $ 343,891.08860759495
# Test Drive Car          $ 954,293.9411764706
# Third Owner             $ 269,474.0032894737
####____________________________________________________________









######___________________________ fuel & km_driven
#_ [201650.5 1] km_driven $ 505,870.31053000235
# CNG # 40 # 277,174.925
# Diesel # 2102 # 675,627.9410085633
# Electric # 1 # 310,000.0
# LPG # 23 # 167,826.04347826086
# Petrol # 2117 # 345,401.8006613132

#_ [403300.0 201650.5] km_driven $ 364,629.5
# CNG # 0 # nan
# Diesel # 49 # 388979.44897959183
# Electric # 0 # nan
# LPG # 0 # nan
# Petrol # 5 # 126,000.0

#_ [604949.5 403300.0] km_driven $ 665,000.0
# CNG # 0 # nan
# Diesel # 2 # 665000.0
# Electric # 0 # nan
# LPG # 0 # nan
# Petrol # 0 # nan

#_ [806599 604949.5] km_driven $250,000.0
# CNG # 0 # nan
# Diesel # 0 # nan
# Electric # 0 # nan
# LPG # 0 # nan
# Petrol # 1 # 250,000.0
####____________________________________________________________






####____________________________________________________________ transmission & km_driven
#print(np.mean(y))
#print(y[tr == 'Automatic'].count())
#print(y[tr == 'Manual'].count())

#_ [806599 : 1] km_driven
# auto :>>   448  # 504,127.3117511521 - 1,408,154.0
# manual :>> 3892 # 504,127.3117511521 - 400,066.6857656732

#_ [201,650.5 : 1] km_driven
# auto :>>   446  # 505,870.31053000235 - 1,412,955.139013453
# manual :>> 3837 # 505,870.31053000235 - 400,433.8149596039

#_ [403,300.0 201,650.5] km_driven
# auto :>>   2  # 364,629.5 - 337,500.0
# manual :>> 52 # 364,629.5 - 365,672.9423076923

#_ [604949.5 403300.0] km_driven
# auto :>>   0  # 665,000.0 - 
# manual :>> 2 # 665,000.0 - 665,000.0

#_ [806599 604949.5] km_driven
# auto :>>   0  # 250,000.0 - 
# manual :>> 1  # 250,000.0 - 250,000.0
####____________________________________________________________







####____________________________________________________________ transmission & fuel
#t = tr[f == 'CNG']; yt = y[f == 'CNG'] 
#print(yt[ t == 'Manual'].count(), yt[ t == 'Automatic'].count(),sep=' ----')
#print(np.mean(yt[ t == 'Manual']), np.mean(yt[ t == 'Automatic']),sep=' ----', end='\n\n')

#t = tr[f == 'Diesel']; yt = y[f == 'Diesel']
#print(yt[ t == 'Manual'].count(), yt[ t == 'Automatic'].count(),sep=' ----')
#print(np.mean(yt[ t == 'Manual']), np.mean(yt[ t == 'Automatic']),sep=' ----', end='\n\n')

#t = tr[f == 'Electric']; yt = y[f == 'Electric']
#print(yt[ t == 'Manual'].count(), yt[ t == 'Automatic'].count(),sep=' ----')
#print(np.mean(yt[ t == 'Manual']), np.mean(yt[ t == 'Automatic']),sep=' ----', end='\n\n')

#t = tr[f == 'LPG']; yt = y[f == 'LPG']
#print(yt[ t == 'Manual'].count(), yt[ t == 'Automatic'].count(),sep=' ----')
#print(np.mean(yt[ t == 'Manual']), np.mean(yt[ t == 'Automatic']),sep=' ----', end='\n\n')

#t = tr[f == 'Petrol']; yt = y[f == 'Petrol']
#print(yt[ t == 'Manual'].count(), yt[ t == 'Automatic'].count(),sep=' ----')
####____________________________________________________________











####____________________________________________________________ fuel_transmission & km_driven

# [ 'CNG'  'LPG'  'Electric'  'Diesel & Automatic'   'Diesel & Manual'   'Petrol & Automatic'   'Petrol & Manual' ]
# [   0      1         2              3                      4                   5                      6         ]

#_ [201650.5 1] km_driven $ 505,870.31053000235

# 0 # 40 # 277,174.925
# 1 # 23 # 167,826.04347826086
# 2 # 1 # 310,000.0
# 3 # 252 # 1,912,503.9603174604 D % A
# 4 # 1850 # 507,145.3697297297  D % M
# 5 # 193 # 766,409.2953367876   P % A
# 6 # 1924 # 303,169.7598752599  P % M

#_ [403300.0 201650.5] km_driven $ 364,629.5
# 0 # 0 # nan
# 1 # 0 # nan
# 2 # 0 # nan
# 3 # 2 # 337,500.0               D % A
# 4 # 47 # 391,170.06382978725    D % M
# 5 # 0 # nan                     P % A
# 6 # 5 # 126000.0                P % M

#_ [604949.5 403300.0] km_driven $ 665,000.0
# 0 # 0 # nan
# 1 # 0 # nan
# 2 # 0 # nan
# 3 # 0 # nan        D % A
# 4 # 2 # 665000.0   D % M
# 5 # 0 # nan        P % A
# 6 # 0 # nan        P % M

#_ [806599 604949.5] km_driven $250,000.0
# 0 # 0 # nan
# 1 # 0 # nan
# 2 # 0 # nan
# 3 # 0 # nan       D % A
# 4 # 0 # nan       D % M
# 5 # 0 # nan       P % A
# 6 # 1 # 250000.0  P % M
####____________________________________________________________