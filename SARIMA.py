#Dataset used is Data_new.csv for 2011 and 2012 and Data1.csv for 2013 onwards
# Importing Required libraries

import numpy as np
#import sklearn as sk
import pandas as pd
import statsmodels as sm
from datetime import datetime
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
#rcParams['figure.figsize'] = 15, 6

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
    ts1 = ts.values.reshape(ts.shape[0],1)
    rolmean = pd.rolling_mean(ts1, window=25)
    rolstd = pd.rolling_std(ts1, window=25)
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
#print(dagra2)

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


#cut value  =  first index of test set after partition!
#end value = total no of months in the dataset
cut = 75
end = 89

#Trying out exponential average technique - not useful!
#expwighted_avg = pd.ewma(monthly_agra, halflife=12)
'''
plt.plot(monthly_agra)
plt.plot(expwighted_avg, color='red')
plt.show()
'''



#To build an ARIMA model - put(0,0,0,0) for seasonal order!
#Note that using high values of period(here,period is 12 for 12 months) makes
#computatuon very expensive and might not even be possible do on a laptop! Thus,
#SARIMA should be run only on monthly data, whereas ARIMA can be run on daily. 
model = SARIMAX(monthly_agra[0:cut,0],order=(2,1,2),seasonal_order=(1,0,0,12),enforce_invertibility=False,enforce_stationarity=False)
result = model.fit(disp=-1)
predict = result.forecast(end-cut)
plt.plot(monthly_agra[cut:end,0])
plt.plot(predict,color='red')
plt.show()
RMSE_12 = RMSE(monthly_agra[cut:end,0],predict)
print(RMSE_12)

