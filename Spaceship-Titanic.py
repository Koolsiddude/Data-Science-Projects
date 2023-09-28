# Databricks notebook source
# MAGIC %run ./Kaggle_API

# COMMAND ----------

competition_name = "spaceship-titanic"
file_path = "/FileStore/tables/Kaggle_Competitions/"

# COMMAND ----------

extract_files(competition_name,file_path)

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/Kaggle_Competitions/spaceship-titanic

# COMMAND ----------

st = spark.read.option("header","true").csv("/FileStore/tables/Kaggle_Competitions/spaceship-titanic/train.csv").toPandas()
st_clone = spark.read.option("header","true").csv("/FileStore/tables/Kaggle_Competitions/spaceship-titanic/train.csv").toPandas()

# COMMAND ----------

display(st)

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

st.head()

# COMMAND ----------

st.isnull().sum(axis=0)

# COMMAND ----------

st_clone.dropna(inplace=True)

# COMMAND ----------

display(st_clone)

# COMMAND ----------

st_clone.isnull().sum(axis=0)

# COMMAND ----------

st_clone.HomePlanet.value_counts()

# COMMAND ----------

st_clone["HomePlanet"] = st_clone["HomePlanet"].map({"Europa": 1,"Earth": 2,"Mars": 3})
st_clone.HomePlanet[:10]

# COMMAND ----------

st.Destination.value_counts()

# COMMAND ----------

st_clone["Destination"] = st_clone["Destination"].map({"TRAPPIST-1e" : 1,"55 Cancri e": 2,"PSO J318.5-22": 3})
st_clone.Destination[:10]

# COMMAND ----------

st_clone.drop(["Name","PassengerId","Cabin"], axis = "columns", inplace=True)

# COMMAND ----------

st_clone.head()

# COMMAND ----------

def to_0_or_1(x):
    if x == "False":
        return 0
    else:
        return 1

# COMMAND ----------

st_clone["CryoSleep"] = st_clone["CryoSleep"].apply(to_0_or_1)
st_clone["VIP"] = st_clone["VIP"].apply(to_0_or_1)

# COMMAND ----------

st_clone.display()

# COMMAND ----------

x_train = st_clone.drop("Transported", axis = 1)
y_train = st_clone["Transported"]

# COMMAND ----------

stt = spark.read.option("header","true").csv("/FileStore/tables/Kaggle_Competitions/spaceship-titanic/test.csv")

# COMMAND ----------

stt = stt.toPandas()

# COMMAND ----------

stt_clone = stt

# COMMAND ----------

stt.display()

# COMMAND ----------

PassengerId = stt_clone["PassengerId"].values
len(PassengerId) 

# COMMAND ----------

def process_data(data):
    data['HomePlanet'] = data['HomePlanet'].map({"Europa": 1,"Earth": 2, "Mars": 3})
    data.Destination = data.Destination.map({"TRAPPIST-1e": 1,"55 Cancri e": 2, "PSO J318.5-22": 3})
    data.drop("Cabin", axis = 1, inplace = True)
    data["CryoSleep"] = data["CryoSleep"].apply(to_0_or_1)
    data["VIP"] = data["VIP"].apply(to_0_or_1)
    return data

# COMMAND ----------

stt_clone = process_data(stt_clone)

# COMMAND ----------

stt_clone.display()

# COMMAND ----------

stt_clone.drop("PassengerId", axis= 1, inplace=True)

# COMMAND ----------

stt_clone.isnull().sum()

# COMMAND ----------

stt_clone.display()

# COMMAND ----------

stt_clone['HomePlanet'].fillna(stt_clone['HomePlanet'].mode()[0],inplace=True)
stt_clone['CryoSleep'].fillna(stt_clone['CryoSleep'].mode()[0],inplace=True)
stt_clone["Destination"].fillna(stt_clone["Destination"].mode()[0],inplace=True)
stt_clone["VIP"].fillna(stt_clone["VIP"].mode()[0],inplace = True)

# COMMAND ----------

stor = stt_clone["Age"].astype(float)

# COMMAND ----------

age_mean = stor.mean()

# COMMAND ----------

stt_clone["Age"] = stt_clone["Age"].astype(float)

# COMMAND ----------

stt_clone["Age"].fillna(stt_clone["Age"].mean(), inplace = True)

# COMMAND ----------

stt_clone["RoomService"] = stt_clone["RoomService"].astype(float)
stt_clone["FoodCourt"] = stt_clone["FoodCourt"].astype(float)
stt_clone["ShoppingMall"] = stt_clone["ShoppingMall"].astype(float)
stt_clone["Spa"] = stt_clone["Spa"].astype(float)
stt_clone["VRDeck"] = stt_clone["VRDeck"].astype(float)

# COMMAND ----------

stt_clone["RoomService"].fillna(stt_clone["RoomService"].mean(), inplace = True)
stt_clone["FoodCourt"].fillna(stt_clone["FoodCourt"].mean(), inplace = True)
stt_clone["ShoppingMall"].fillna(stt_clone["ShoppingMall"].mean(), inplace = True)
stt_clone["Spa"].fillna(stt_clone["Spa"].mean(),inplace = True)
stt_clone["VRDeck"].fillna(stt_clone["VRDeck"].mean(), inplace = True)

# COMMAND ----------

stt_clone.isnull().sum()

# COMMAND ----------

stt_clone.display()

# COMMAND ----------

print(type(stt_clone.Age))

# COMMAND ----------

stt_clone.info()

# COMMAND ----------

stt_clone.drop("Name", axis=1,inplace=True)

# COMMAND ----------

stt_clone.skew()

# COMMAND ----------

x_test=stt_clone.iloc[:,0:]

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

# COMMAND ----------

lg_model = LogisticRegression()
lg_model.fit(x_train,y_train)

# COMMAND ----------

st_prediction = lg_model.predict(stt_clone)

# COMMAND ----------

display(st_prediction)

# COMMAND ----------

lg_model.score(x_test,st_prediction)

# COMMAND ----------

x_test.shape

# COMMAND ----------

data=pd.DataFrame(st_prediction,columns=["Transported"])

# COMMAND ----------

data1=pd.DataFrame(PassengerId, columns=["PassengerId"])

# COMMAND ----------

result=pd.concat([data1,data],axis=1)

# COMMAND ----------

result
