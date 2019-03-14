# Importing Required libraries

import numpy as np
import sklearn as sk
import pandas as pd
import statsmodels as sm
from datetime import datetime
from matplotlib import pyplot
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

#Loading the datasets

dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
data1 = pd.read_csv('Data1.csv', parse_dates=['Price Date'], index_col=['Price Date'],date_parser=dateparse)
#data1 = pd.read_csv('Data1.csv')
dagra = data1[data1['District Name']=='Bangalore']
#print(dagra)
#print(data1.index[6])

def RMSE(actual,predicted):
    MSE_error=np.matmul((actual-predicted).T,(actual-predicted))
    MSE_error=np.divide(MSE_error,predicted.shape[0])
    MSE_error=np.sqrt(MSE_error)
    return MSE_error



'''
def stationarity_test(ts):
    #Determing rolling statistics
    ts1 = ts.reshape(ts.shape[0],1)
    rolmean = pd.rolling_mean(ts1, window=3)
    rolstd = pd.rolling_std(ts1, window=3)
    #print(rolmean)
    #Plot rolling statistics:
    orig = plt.plot(ts1, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Dicky-Fuller Test:- 
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
'''



data2 = pd.read_csv('Data_new.csv')
dagra2 = data2[data2['District']=='Bangalore']
dagra2012 = dagra2[dagra2['Year']==2012]
dagra2011 = dagra2[dagra2['Year']==2011]
#print(dagra2)


## Monthly - Hybrid
monthly_agra = np.zeros((161,1))
for i in range(0,96):
    year = 2005 + (i-i%12)/12
    temp = dagra2[dagra2['Year']==year]
    month = 1+i%12
    temp1 = temp[temp['Month']==month]
    monthly_agra[i] = np.mean(temp1.iloc[:,2])

for i in range(96,161):
    j = i-96
    year = str(2013+int((j-j%12)/12))
    month = str(1+j%12)
    d=dagra[year + '-' + month]
    monthly_agra[i] = np.mean(d.iloc[:,2])



'''
#Bollinger band type smoothening
rolmean = pd.rolling_mean(monthly_agra, window=3)
rolstd = pd.rolling_std(monthly_agra, window=3)
#plt.plot(rolstd,color='red')
new = np.zeros((89,1))
rolmean_new = np.zeros((89,1))
rolstd_new = np.zeros((89,1))
new[0] = monthly_agra[0]
new[1] = monthly_agra[1]
new[2] = monthly_agra[2]
rolmean_new[0] = rolmean[0]
rolmean_new[1] = rolmean[1]
rolmean_new[2] = rolmean[2]
rolstd_new[0] = rolstd[0]
rolstd_new[1] = rolstd[1]
rolstd_new[2] = rolstd[2]

for i in range(3,89):
    if(monthly_agra[i]>(rolmean_new[i-1]+2*rolstd_new[i-1])):
       new[i] = (rolmean_new[i-1]+2*rolstd_new[i-1])
    else:
        new[i] = monthly_agra[i]
    rolmean_new[i] = (new[i]+new[i-1]+new[i-2])/3
    rolstd_new[i] = np.sqrt(((new[i]-rolmean_new[i])*(new[i]-rolmean_new[i])+(new[i-1]-rolmean_new[i])*(new[i-1]-rolmean_new[i])+(new[i-2]-rolmean_new[i])*(new[i-2]-rolmean_new[i]))/2)
'''


#cut is cut point in reference to indices of prices.
#same goes for end
cut = 120
end = monthly_agra.shape[0]                      


#SARIMAX, Residuals Calculation-
model = SARIMAX(monthly_agra[0:cut,0],order=(2,1,2),seasonal_order=(2,1,1,12),enforce_invertibility=False,enforce_stationarity=False)
result = model.fit(disp=-1)
fitted = result.fittedvalues
predict_SARIMAX = result.forecast(end-cut)
predict_SARIMAX = predict_SARIMAX.reshape(end-cut,1)
fitted[0] = monthly_agra[0,0]
fitted = fitted.reshape(cut,1)

residuals = monthly_agra[0:cut,:]-fitted
residuals = residuals.reshape(cut,1)
SARIMAX_pred = np.concatenate((fitted,predict_SARIMAX),axis=0)
SARIMAX_pred = SARIMAX_pred.reshape(end,1)


#Preparing Data for Neural Network - 

##In Neural Network - inputs for predicting ith are - S(SARIMAX value),R(residual)
#For reg part - S[i],S[i-1],S[i-2],...S[i-reg],R[i-1],R[i-2],...R[i-reg]
#For seas par - S[i-12],S[i-13],...S[i-(12+seas-1)],R[i-12],R[i-13],...R[i-(12+seas-1)]

reg = 2
seas = 2
period = 12
#start = 2
start = period + seas - 1
XNN = np.zeros((cut-start,2*(reg+seas)+1))
Labels = np.zeros((XNN.shape[0],1))


for i in range(0,XNN.shape[0]):
    for j in range(0,reg+1):
        XNN[i,j] = SARIMAX_pred[i+start-j]
    for j in range(reg+1,2*reg+1):
        XNN[i,j] = residuals[i+start-j+reg]
    for j in range(2*reg+1,2*reg+seas+1):
        k = j-2*reg-1+period
        XNN[i,j] = SARIMAX_pred[i+start-k]
    for j in range(2*reg+seas+1,2*(reg+seas)+1):
        k = j-(2*reg+seas+1)+period
        XNN[i,j] = residuals[i+start-k]
    Labels[i] = residuals[i+start]

Labels = Labels.reshape(XNN.shape[0],)



model = MLPRegressor(hidden_layer_sizes=(50,50,50),max_iter=200000,random_state=5,activation = 'tanh',learning_rate_init = 0.0001,learning_rate='adaptive')
result = model.fit(XNN,Labels)

residuals_pred = np.zeros((end-cut,1))

for i in range(0,end-cut):
    X = np.array([SARIMAX_pred[i+cut],SARIMAX_pred[i+cut-1],SARIMAX_pred[i+cut-2],residuals[i+cut-1],residuals[i+cut-2],SARIMAX_pred[i+cut-period],SARIMAX_pred[i+cut-period-1],residuals[i+cut-period],residuals[i+cut-period-1]])
    residuals_pred[i,0] = result.predict(X.reshape(1,-1))
    residuals = np.append(residuals,residuals_pred[i,0])

#SARIMAX_pred[i+cut],SARIMAX_pred[i+cut-1],SARIMAX_pred[i+cut-2],residuals[i+cut-1],residuals[i+cut-2]
residuals = residuals.reshape(end,1)
predict = SARIMAX_pred[cut:end,0] + residuals[cut:end,0]
predict = predict.reshape(end-cut,1)
plt.plot(monthly_agra[:,0])
plt.plot(SARIMAX_pred[:,0],color='red')
#plt.plot(predict,color='green')
#plt.plot(residuals)
plt.show()
print(RMSE(monthly_agra[cut:end,:],predict))
#print(SMAPE(monthly_agra[cut:end,:],predict))


## Daily Hybrid -

'''
daily = dagra2011.iloc[:,2]
daily = daily.reshape(287,1)
daily = np.concatenate((daily,dagra2012.iloc[:,2].reshape(286,1)),axis=0)
daily = np.concatenate((daily,dagra.iloc[:,2].reshape(1468,1)),axis=0)
'''

'''
daily = dagra2.iloc[:,2]
daily = daily.reshape(daily.shape[0],1)
X = dagra.iloc[:,2]
daily = np.concatenate((daily,X.reshape(X.shape[0],1)),axis=0)
plt.plot(daily)
plt.show()



cut = 1800
end = daily.shape[0]                      


#SARIMAX, Residuals Calculation-
model = SARIMAX(daily[0:cut,0],order=(6,1,2),seasonal_order=(0,0,0,0),enforce_invertibility=False,enforce_stationarity=False)
result = model.fit(disp=-1)
fitted = result.fittedvalues
predict_SARIMAX = result.forecast(end-cut)
predict_SARIMAX = predict_SARIMAX.reshape(end-cut,1)
fitted[0] = daily[0,0]
fitted = fitted.reshape(cut,1)

residuals = daily[0:cut,:]-fitted
residuals = residuals.reshape(cut,1)
SARIMAX_pred = np.concatenate((fitted,predict_SARIMAX),axis=0)
SARIMAX_pred = SARIMAX_pred.reshape(end,1)
plt.plot(daily)
plt.plot(SARIMAX_pred,color='red')
plt.show()
'''

