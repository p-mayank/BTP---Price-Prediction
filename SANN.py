# Importing Required libraries

import numpy as np
import sklearn as sk
import pandas as pd
import statsmodels as sm
from datetime import datetime
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neural_network import MLPRegressor

#Loading the datasets

dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
data1 = pd.read_csv('Data1.csv', parse_dates=['Price Date'], index_col=['Price Date'],date_parser=dateparse)
#data1 = pd.read_csv('Data1.csv')
dagra = data1[data1['District Name']=='Agra']
#print(dagra)
#print(data1.index[6])

def RMSE(actual,predicted):
    MSE_error=np.matmul((actual-predicted).T,(actual-predicted))
    MSE_error=np.divide(MSE_error,predicted.shape[0])
    MSE_error=np.sqrt(MSE_error)
    return MSE_error


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






data2 = pd.read_csv('Data_new.csv')
dagra2 = data2[data2['District']=='Agra']
dagra2012 = dagra2[dagra2['Year']==2012]
dagra2011 = dagra2[dagra2['Year']==2011]
#print(dagra2)

#Monthly SANN (Seasonal Artificial Neural Network) - 

#Following 2 for loops are used to convert Daily data to monthly data!
#Thus, highly dependent on data format; If monthly data is available already,
#directly put it in SARIMAX function!

monthly_agra = np.zeros((89,1))
for i in range(0,24):
    year = 2011 + (i-i%12)/12
    temp = dagra2[dagra2['Year']==year]
    month = 1+i%12
    temp1 = temp[temp['Month']==month]
    monthly_agra[i] = np.mean(temp1.iloc[:,2])

for i in range(24,89):
    j = i-24
    year = str(2013+int((j-j%12)/12))
    month = str(1+j%12)
    d=dagra[year + '-' + month]
    monthly_agra[i] = np.mean(d.iloc[:,2])



'''
#Trying out Bollinger band type smoothening - i.e. if replacing a spike
#outside bollinger band by moving average. 
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

#no of past values = reg
#no of past season values = seas
reg = 3
seas = 3
period = 12
#start = 2
start = period + seas - 1
XNN = np.zeros((monthly_agra.shape[0]-start,reg+seas))
Labels = np.zeros((XNN.shape[0],1))

cut = 80
end = monthly_agra.shape[0]
#Next few lines for - Trying first order differencing and logarithmic compression
shifted = monthly_agra[0:end-1,:]
shifted = np.insert(shifted,0,monthly_agra[0,0])
shifted = shifted.reshape(end,1)
#monthly_agra_diff = np.log(monthly_agra) - np.log(shifted)
#monthly_agra_diff = np.log(monthly_agra)
monthly_agra_diff = monthly_agra
#plt.plot(monthly_agra)
#plt.show()


for i in range(0,XNN.shape[0]):
    for j in range(0,reg):
        XNN[i,j] = monthly_agra_diff[i+start-1-j]
    for j in range(reg,reg+seas):
        XNN[i,j] = monthly_agra_diff[i+reg+seas-1-j]
    Labels[i] = monthly_agra_diff[i+start]

#print(monthly_agra[25]," ",monthly_agra[24]," ",monthly_agra[13]," ",monthly_agra[12])
#print(XNN[12,:])

#cut is cut point in reference to indices of prices.
#same goes for end


monthly = monthly_agra_diff[0:cut,0]
XNN_new = XNN[0:cut-start,:]
Labels_new = Labels[0:cut-start,0]

##(50,50,100) ; (10,50,10) - log differencing(reg=0:2,seas=0:2); - activaation - tanh/logistic
##(50,50,50)-just log(reg=2,seas=1) - BEST OF ALL(RMSE~90-95) - activaation - relu
##Plain modelling(TRY other orders) - (50,50,50)(reg=1,seas=1) - RMSE~125 or more - activaation - relu
model = MLPRegressor(hidden_layer_sizes=(50,50,50),max_iter=200000,random_state=5,activation = 'relu',learning_rate_init = 0.0001,learning_rate='adaptive')
result = model.fit(XNN_new,Labels_new)
pred1 = result.predict(XNN[cut-start:end-start,:])
pred1 = pred1.reshape(end-cut,1)

pred = np.zeros((end-cut,1))
for i in range(0,end-cut):
    '''
    if(monthly[i+cut+start-period]>rolmean[i+cut+start-period]+2*rolstd[i+cut+start-period]):
        a = new[i+cut+start-period]
    else:
        a = monthly[i+cut+start-period]
    '''
    # Make X according to the values taken by seas,reg and the arrangement of XNN!
    X = np.array([(monthly[i+cut-1],monthly[i+cut-2],monthly[i+cut-3],monthly[i+cut-period],monthly[i+cut-period-1],monthly[i+cut-period-2])])
    pred[i,0] = result.predict(X)
    monthly = np.append(monthly,pred[i,0])

#,monthly[i+cut-3],monthly[i+cut-4],monthly[i+cut-5],monthly[i+cut-6],monthly[i+cut-7],monthly[i+cut-8],monthly[i+cut-9],monthly[i+cut-10],monthly[i+cut-11],monthly[i+cut-12]




plt.plot(monthly_agra[cut:end,:])
plt.plot(pred, color='red')
plt.show()
print(RMSE(monthly_agra[cut:end,:],pred))


#Below model is for daily data - 
'''
#Daily TDNN model
#print(dagra['2017'])
prices = dagra2011.iloc[:,2].values.reshape(287,1)
prices = np.concatenate((prices,dagra2012.iloc[:,2].values.reshape(286,1)),axis=0)
prices = np.concatenate((prices,dagra.iloc[:,2].values.reshape(1468,1)),axis=0)

reg = 7
seas = 1
period = 287
#start = 7
start = period + seas - 1
XNN_daily = np.zeros((prices.shape[0]-start,reg+seas))
Labels_daily = np.zeros((XNN_daily.shape[0],1))

for i in range(0,XNN_daily.shape[0]):
    for j in range(0,reg):
        XNN_daily[i,j] = prices[i+start-1-j]
    for j in range(reg,reg+seas):
        XNN_daily[i,j] = prices[i+reg+seas-1-j]
    Labels_daily[i] = prices[i+start]

#cut is cut point in reference to indices of prices.
#same goes for end
cut = 1900
end = prices.shape[0]
daily = prices[0:cut,0]
XNN_dnew = XNN_daily[0:cut-start,:]
Labels_dnew = Labels_daily[0:cut-start,0]

model_d = MLPRegressor(hidden_layer_sizes=(3,),max_iter=50000,activation = 'relu',learning_rate_init = 0.001,learning_rate='adaptive')
result_d = model_d.fit(XNN_dnew,Labels_dnew)
pred1_d = result_d.predict(XNN_daily[cut-start:end-start,:])
pred1_d = pred1_d.reshape(end-cut,1)

pred_d = np.zeros((end-cut,1))
for i in range(0,end-cut):
    X_d = np.array([(daily[i+cut-1],daily[i+cut-2],daily[i+cut-3],daily[i+cut-4],daily[i+cut-5],daily[i+cut-6],daily[i+cut-7],daily[i+cut-period])])
    pred_d[i,0] = result_d.predict(X_d)
    daily = np.append(daily,pred_d[i,0])

#X_d = np.array([(daily[i+cut-1],daily[i+cut-2],daily[i+cut-3],daily[i+cut-4],daily[i+cut-5],daily[i+cut-6],daily[i+cut-7],daily[i+cut-period],daily[i+cut-period-1],daily[i+cut-period-2],daily[i+cut-period-3],daily[i+cut-period-4],daily[i+cut-period-5],daily[i+cut-period-6])])

plt.plot(Labels_daily[cut-start:end-start,:])
plt.plot(pred_d, color='red')
plt.show()
print(RMSE(Labels_daily[cut-start:end-start,:],pred_d))
'''
