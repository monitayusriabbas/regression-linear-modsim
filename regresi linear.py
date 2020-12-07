#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[7]:


#masukkan path dimana lokasi file berada
path = 'https://raw.githubusercontent.com/monitayusriabbas/regression-linear-modsim/main/china_gdp.csv'
df = pd.read_csv(path)


# In[8]:


df = pd.read_csv(path)

#untuk lihat isi 10 data dari file
df.head(10)


# In[17]:


#plotting dataset
plt.figure(figsize=(7,5))
x_data, y_data = (df["Year"].values, df["Value"].values )
plt.plot(x_data, y_data, "ro")
plt.ylabel("GDP")
plt.xlabel("Tahun")

#untuk menampilkan visualisasi dari dataset
plt.show()


# In[18]:


# melakukan normalisasi data x
dataX = x_data/max(x_data)
dataY = y_data/max(y_data)

#menentukan parameter
#pcov : parameter covarience
popt, pcov = curve_fit(sigmoid, dataX, dataY)

# menlihat nilai popt:
print(popt)


# In[19]:


#membuat model sigmoid
def sigmoid(x, beta_1, beta_2):
    si = 1/(1+np.exp(-beta_1*(x-beta_2)))
    return si


# In[22]:


# normalisasi
x = x_data/max(x_data)
# labelfit
y = sigmoid(x, *popt)

# tampilan hasil regresi non linear
plt.figure(figsize=(7, 5))
plt.plot(x_data, dataY, 'ro', label='data')
plt.plot(x_data, y, label='fit')
plt.legend(loc='best')
plt.xlabel('Tahun')
plt.ylabel('GDP')
plt.show()


# In[23]:


# generate angka tahun dari 2015 - 2030
x_2015_2030 = np.linspace(2015, 2030, num=16)
# normalisasi x_2015_2030
x_2015_2030_norm = x_2015_2030/max(x_data)
# fit
y = sigmoid(x_2015_2030_norm, *popt)


# In[24]:


plt.figure(figsize=(8, 5))
plt.plot(x_2015_2030, y, label='prediksi')
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()


# In[33]:


# data prediksi
y = sigmoid(dataX, *popt)

# nilai MAE
print(mean_absolute_error(dataY, y))

# nilai MSE
print(mean_squared_error(dataY, y))

# nilai R2
print(r2_score(dataY, y))


# In[ ]:




