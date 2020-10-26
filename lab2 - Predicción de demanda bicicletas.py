#!/usr/bin/env python
# coding: utf-8

# ### Laboratorio 2 – Proyecto de Predicción de Predicción de Bicicletas
# 
# En esta sección ud. estará a cargo de evaluar como se puede realizar la predicción de bicicletas de un local de alquiler.  Como científico de datos tiene la labor de predecir la siguiente demanda del local por medio de un modelo de machine learning utilizando scikit-learn.

# ### Copiando el Dataset al Computador

# Primero debemos adquirir el dataset.  Nuestro dataset a descargar está en la siguiente dirección:
# - https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip

# ### Cargar y explorar el dataset
# 
# Ahora cargue el dataset y explore las diferentes variables del dataset:
# - hour.csv

# Cargar el dataset con pandas

# In[ ]:


#Instalando librerias - Descomentar para ejecutar

'''
!pip install xgboost
!pip install sklearn
!pip install matplotlib
'''


# In[1]:


import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# In[2]:


bikeshare = pd.read_csv('hour.csv')


# Explorar las primeras 48 filas

# In[3]:


bikeshare.head(48)


# ### Desplegar información del dataset
# 
# Despliegue información escencial del dataset

# -Descripción del dataset : count de valores, media, desviación standar, valor min/max, percentiles 25,50,75

# In[4]:


bikeshare.describe()


# -Tipo de datos por cada columna en el dataset para evaluar si tenemos que hacer alguna transformación

# In[5]:


bikeshare.dtypes


# In[6]:


bikeshare.season.value_counts()


# In[7]:


bikeshare.hr.value_counts()


# In[8]:


bikeshare.holiday.value_counts()


# In[9]:


bikeshare.weathersit.value_counts()


# ### Visualizando el dataset

# In[10]:


columnas = bikeshare.columns


# In[11]:


columnas


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot


# SU CODIGO AQUI
bikeshare.hist(figsize=(30,15));


# ### Variables Dummy - Transformación de Data Categórica
# 
# Los datos categóricos como temporada, mes, y año deben ser transformados a números debido a que los modelos solo trabajan con esto.  También tendremos variables binarias.  La acción la realizaremos com pandas y la función:
#  - get_dummies()
#  - columnas dummy = 'season', 'weathersit', 'mnth', 'hr', 'weekday'
#  - columnas a borrar = 'instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr'

# In[13]:


dummy_cols = [ 'season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_cols:
    dummies = pd.get_dummies(bikeshare[each], prefix=each, drop_first=False)
    bikeshare = pd.concat([bikeshare, dummies], axis=1)

drop_cols = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = bikeshare.drop(drop_cols, axis=1)
data.head()


# In[14]:


data.columns


# ### Escalar el Dataset
# - variables a escalar = 'casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed'

# In[15]:


to_scale = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in to_scale:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
    


# In[33]:


data


# In[16]:


data.columns


# ### Separación del Dataset
# Separar el dataset en Training y Test
# - 20% test set
# - Campos de Test = 'cnt', 'casual', 'registered'
# - Campos de Train = todos los demas

# In[17]:


# Separate the data into features and targets
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.20, random_state=42)

X_train = train.drop(['cnt', 'casual', 'registered'], axis=1)
X_test = test.drop(['cnt', 'casual', 'registered'], axis=1)
y_train = train['cnt']
y_test = test['cnt']


# ### Entrenamiento
# 
# Entrenar el modelo basado en el regresor XGBRegressor

# In[18]:


from sklearn import neighbors
from sklearn.metrics import mean_squared_error 

xgbr = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1) 
print(xgbr)


# In[19]:


xgbr.fit(X_train, y_train)


# In[20]:


score = xgbr.score(X_train, y_train)
print("Training score: ", score)


# #### Mean Cross Validation

# In[21]:


scores = cross_val_score(xgbr, X_train, y_train,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


# #### K-Fol Cross Validation

# In[22]:


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# ### Guardar el Modelo 
# 
# Una parte importante es que luego de entrenar podríamos guardar el modelo.  Este paso se realiza luego de haber evaluado varios y validar que cumple con las métricas.  A manera de prueba lo realizaremos antes, sin embargo recordar que es parte de los últimos pasos.

# In[23]:


# Guardar el modelo
import pickle

# Guardando el modelo en disco
filename = 'modelo_final.sav'
pickle.dump(xgbr, open(filename, 'wb'))


# ### Cargar el Modelo
# 
# Cuando poseemos la aplicación final, por ejemplo una aplicación por celular o una aplicación web, podemos 'consumir' el modelo llamandolo para realizar nuestras predicciones.  Para esto el modelo debe estar disponible desde una ubicación conocida en la máquina o dispositivo a ejecutar.

# In[24]:


# Cargar el modelo
loaded_model = pickle.load(open(filename, 'rb'))


# In[25]:


result = loaded_model.score(X_test, y_test)
print(result)


# ### Realizar predicciones
# 
# Realizar las predicciones con el modelo de los datos de test

# In[26]:


# SU CODIGO AQUI
y_pred = loaded_model.predict(X_test)


# In[34]:


X_test


# In[27]:


#predicciones
y_pred


# ### Evaluar el modelo por medio de métricas
# 
# Evaluar el modelo por medio de las métricas de error medio cuadrado

# In[28]:


mse = mean_squared_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse*(1/2.0)))


# ### Graficar los resultados
# 
# Graficar los resultados de la predicción vs los resultados de las pruebas

# In[29]:


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Grafico de prediccion - short")
plt.legend()
plt.show()


# In[30]:


fig, ax = plt.subplots(figsize=(30,15))

ax.plot(y_test.values, label='Datos')
ax.plot(y_pred[:], label='Prediccion')
ax.set_xlim(right=len(y_pred))
ax.legend()

dates = pd.to_datetime(bikeshare.loc[y_test.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=90)


# In[31]:


df_test_final = X_test.copy()
df_test_final['cnt_scaled'] = y_test
df_test_final['cnt_scaled_predicted'] = y_pred


# In[32]:


df_test_final.head(10)
#Dataframe con la predicción  como columna final

