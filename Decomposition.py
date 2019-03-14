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




#p = results_ARIMA.forecast(468)
#print(RMSE(dagra.iloc[1000:1468,2],p[0]))

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


##Decomposing Time Series into Trend,Seasonality and Residuals!!
##Implementing using Decomposed Time Series! -
##ARIMA/exp. smoothing for trend, SARIMA for seasonality, NN for residuals or ignoring them!


result = seasonal_decompose(monthly_agra, model='additive',freq=12,two_sided=False)
trend = result.trend
seasonal = result.seasonal
res = result.resid
#result = seasonal_decompose(res[12:], model='multiplicative',freq=12,two_sided=False)


cut = 140
end = monthly_agra.shape[0]                      


#SARIMAX, Residuals Calculation-
trend = trend.reshape(trend.shape[0],1)
model_trend = SARIMAX(trend[0:cut,:],order=(1,1,1),seasonal_order=(0,0,0,0),enforce_invertibility=False,enforce_stationarity=False)
result_trend = model_trend.fit(disp=-1)
predict_trend = result_trend.forecast(end-cut)
predict_trend = predict_trend.reshape(end-cut,1)
fitted_trend = result_trend.fittedvalues
#plt.plot(trend[cut:end,:])
#plt.plot(predict_trend)
#plt.plot(fitted_trend,color='red')
#plt.show()

seasonal = seasonal.reshape(seasonal.shape[0],1)
model_seas = SARIMAX(seasonal[0:cut,:],order=(1,1,1),seasonal_order=(1,1,0,12),enforce_invertibility=False,enforce_stationarity=False)
result_seas = model_seas.fit(disp=-1)
fitted_seas = result_seas.fittedvalues
predict_seas = result_seas.forecast(end-cut)
predict_seas = predict_seas.reshape(end-cut,1)
#plt.plot(seasonal[cut:end,:])
#plt.plot(predict_seas)
#plt.show()
#print(RMSE(seasonal[cut:end,:],predict_seas))



reg = 12
seas = 0
period = 0
start = 24
#start = period + seas - 1
XNN = np.zeros((cut-start,reg))
Labels = np.zeros((XNN.shape[0],1))

res = res.reshape(res.shape[0],1)
resid = res[0:cut,:]
print(resid.shape)

for i in range(0,XNN.shape[0]):
    for j in range(0,reg):
        XNN[i,j] = res[i+start-j-1]
    Labels[i] = res[i+start]

Labels = Labels.reshape(XNN.shape[0],)



model = MLPRegressor(hidden_layer_sizes=(50,50,50),max_iter=200000,random_state=5,activation = 'tanh',learning_rate_init = 0.0001,learning_rate='adaptive')
result = model.fit(XNN,Labels)

predict_res = np.zeros((end-cut,1))

for i in range(0,end-cut):
    X = np.array([resid[i+cut-1],resid[i+cut-2],resid[i+cut-3],resid[i+cut-4],resid[i+cut-5],resid[i+cut-6],resid[i+cut-7],resid[i+cut-8],resid[i+cut-9],resid[i+cut-10],resid[i+cut-11],resid[i+cut-12]])
    predict_res[i,0] = result.predict(X.reshape(1,-1))
    resid = np.append(resid,predict_res[i,0])
#''
res = res.reshape(res.shape[0],1)
model_res = SARIMAX(res[0:cut,:],order=(2,1,2),seasonal_order=(2,1,1,12),enforce_invertibility=False,enforce_stationarity=False)
result_res = model_res.fit(disp=-1)
predict_res = result_res.forecast(end-cut)
predict_res = predict_res.reshape(end-cut,1)
#''

'''
plt.plot(res[cut:end,:])
plt.plot(predict_res)
plt.show()
'''

temp = predict_res + trend[cut:end] + predict_seas
plt.plot(monthly_agra[cut:end,:])
plt.plot(temp)
plt.show()
print(RMSE(monthly_agra[cut:end,:],temp))

#Time Series Decomposition method for Daily Data

'''
daily = dagra2.iloc[:,2]
daily = daily.reshape(daily.shape[0],1)
size = dagra.iloc[:,2].shape[0]
temp = dagra.iloc[:,2]
temp = temp.reshape(size,1)
daily = np.concatenate((daily,temp),axis=0)

result_daily = seasonal_decompose(daily, model='multiplicative',freq=280,two_sided=False)
trend_daily = result_daily.trend
seasonal_daily = result_daily.seasonal
res_daily = result_daily.resid


cut = 3500
end = daily.shape[0]


seasonal_daily = seasonal_daily.reshape(seasonal_daily.shape[0],1)
dmodel_seas = SARIMAX(seasonal_daily[0:cut,:],order=(1,1,1),seasonal_order=(1,1,0,12),enforce_invertibility=False,enforce_stationarity=False)
dresult_seas = dmodel_seas.fit(disp=-1)
dfitted_seas = dresult_seas.fittedvalues
dpredict_seas = dresult_seas.forecast(end-cut)
dpredict_seas = dpredict_seas.reshape(end-cut,1)
#print(RMSE(seasonal_daily[cut:end,:],dpredict_seas))


reg = 3
seas = 0
period = 0
start = 287
#start = period + seas - 1
XNN = np.zeros((cut-start,reg))
Labels = np.zeros((XNN.shape[0],1))

res_daily = res_daily.reshape(res_daily.shape[0],1)
resid = res_daily[0:cut,:]


for i in range(0,XNN.shape[0]):
    for j in range(0,reg):
        XNN[i,j] = res_daily[i+start-j-1]
    Labels[i] = res_daily[i+start]

Labels = Labels.reshape(XNN.shape[0],)



model = MLPRegressor(hidden_layer_sizes=(5,5,5),max_iter=200000,random_state=5,activation = 'logistic',learning_rate_init = 0.0001,learning_rate='adaptive')
result = model.fit(XNN,Labels)

dpredict_res = np.zeros((end-cut,1))

for i in range(0,end-cut):
    X = np.array([resid[i+cut-1],resid[i+cut-2],resid[i+cut-3]])
    dpredict_res[i,0] = result.predict(X.reshape(1,-1))
    resid = np.append(resid,dpredict_res[i,0])

#,resid[i+cut-4],resid[i+cut-5],resid[i+cut-6],resid[i+cut-7]
plt.plot(res_daily[cut:end,:])
plt.plot(dpredict_res)
plt.show()
'''
