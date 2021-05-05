# Importing Libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xg
from sklearn.metrics import accuracy_score,confusion_matrix,auc,f1_score
from sklearn.model_selection import train_test_split,cross_val_score, KFold, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import streamlit as st 

#------------------------------------------------------------------------------------------------------#

# Reading Data

df_train = pd.read_csv(r"..\Titanic\Data\train.csv")
#df_train.head()
df_test = pd.read_csv(r"..\Titanic\Data\test.csv")
#df_test.head()

df_train_org = df_train
df_test_org = df_test

#Deleting the passenget ID from both datasets
df_train = df_train.drop("PassengerId", axis =1)
df_test = df_test.drop("PassengerId", axis =1)

#------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#

# Feature Engineering

# Converting Cabins

df_train["Cabin"].fillna("Not Available", inplace = True)

for i in np.arange(0,890):
    if df_train["Cabin"][i] != "Not Available":
        df_train["Cabin"][i] = "Available"
        
#df_train["Cabin"]

df_test["Cabin"].fillna("Not Available", inplace = True)

for i in np.arange(0,418):
    if df_test["Cabin"][i] != "Not Available":
        df_test["Cabin"][i] = "Available"
        
#df_test["Cabin"]
#------------------------------------------------------------------------------------------------------#

#Converting Ages

df_train["Age"] = df_train["Age"].fillna(df_test["Age"].median())
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())

for i in np.arange(0,891):
    if df_train["Age"][i] <= 15:
        df_train["Age"][i] = "Children"
    elif df_train["Age"][i] > 15 and df_train["Age"][i] <= 30:
        df_train["Age"][i] = "Young Aged"
    elif df_train["Age"][i] > 30 and df_train["Age"][i] <= 45:
        df_train["Age"][i] = "Middle Aged"
    elif df_train["Age"][i] > 45 and df_train["Age"][i] <= 60:
        df_train["Age"][i] = "Senior Aged"
    else:
        df_train["Age"][i] = "Old"

#df_train["Age"]

for i in np.arange(0,418):
    if df_test["Age"][i] <= 15:
        df_test["Age"][i] = "Children"
    elif df_test["Age"][i] > 15 and df_test["Age"][i] <= 30:
        df_test["Age"][i] = "Young Aged"
    elif df_test["Age"][i] > 30 and df_test["Age"][i] <= 45:
        df_test["Age"][i] = "Middle Aged"
    elif df_test["Age"][i] > 45 and df_test["Age"][i] <= 60:
        df_test["Age"][i] = "Senior Aged"
    else:
        df_test["Age"][i] = "Old"   
        
#df_test["Age"]

#------------------------------------------------------------------------------------------------------#

#Combining SibSp and Parch for creating new feature named Family

df_train["Family"] = df_train["SibSp"] + df_train["Parch"] + 1
df_test["Family"] = df_test["SibSp"] + df_test["Parch"] + 1

#------------------------------------------------------------------------------------------------------#

# Filling the rest Nan Values

df_train["Embarked"] = df_train["Embarked"].fillna(df_train["Embarked"].mode())
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())

#------------------------------------------------------------------------------------------------------#
# Using One hot encoding on Categorical Variables

# Dropping Name and Ticket Column

df_train = df_train.drop(columns = ["Name", 'Ticket'])
df_test = df_test.drop(columns = ["Name", 'Ticket'])

cols = ["Pclass", 'Sex', 'Age', 'Cabin','Embarked']
for i in cols:
    df1 = pd.get_dummies(df_train[i], drop_first = True)
    df2 = pd.get_dummies(df_test[i], drop_first = True) 
    df_train = pd.concat([df_train,df1], axis = 1)
    df_test = pd.concat([df_test,df2], axis = 1)

df_train = df_train.drop(columns=["Pclass", 'Sex', 'Age', 'Cabin','Embarked'], axis=1)
df_test = df_test.drop(columns=["Pclass", 'Sex', 'Age', 'Cabin','Embarked'], axis=1)

#df_test.head()

#------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#

# Modeling Building 

def RandomForest(df_train,df_test):
    """
    Performs Random Forest Classifier
    """
    X = df_train.drop("Survived", axis= 1)
    y = df_train["Survived"]
    X_main = df_test
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
    rf = RandomForestClassifier() 
    scoring = ["accuracy", "f1"]
    
    param_grid = {
        
        'n_estimators': [10, 30, 50, 100, 250, 500],
        'max_features': [1, 2, 3, 4],
        'min_samples_split': [5, 10, 15, 20, 30],
        'max_depth': [6, 9, 15, 25],
        
    }
    
    search = GridSearchCV(rf, param_grid, n_jobs=-1, return_train_score= True)
    search.fit(X_train, y_train)
    
    scores_rf = cross_validate(search.best_estimator_, X_train, y_train, cv=5, 
                        scoring = scoring, return_train_score= True)
    
    y_rf = search.best_estimator_.predict(X_test)
    y_rf_train = search.best_estimator_.predict(X_train)
    y_rf_main = search.best_estimator_.predict(X_main)
    
    metrics_train = [scores_rf['train_accuracy'].mean(),
                scores_rf['train_accuracy'].std(),
                scores_rf["train_f1"].mean(),
                scores_rf["train_f1"].std()]
    
    metrics_valid = [scores_rf['test_accuracy'].mean(),
                scores_rf['test_accuracy'].std(),
                scores_rf["test_f1"].mean(),
                scores_rf["test_f1"].std()]
    
    metrics_test = [accuracy_score(y_test, y_rf),f1_score(y_test, y_rf)]
    
    arrays = [
        np.array([ "Accuracy", "Accuracy", "F1-Score", "F1-Score"]),
        np.array(["Mean", "Std", "Mean", "Std"])
    ]
    
    table_cv_rf = pd.DataFrame({"Training": metrics_train,"Validation": metrics_valid},index = arrays)
    table_test_rf = pd.DataFrame({"Testing": metrics_test}, index = ["Accuracy", "F1-score"])
    
    y_rf_train_s = pd.Series(y_rf_train)
    y_rf_s = pd.Series(y_rf)
    
    my_dict = dict(A = y_rf_train_s,
                   B = y_rf_s)
    
    phi_rf = pd.DataFrame.from_dict(my_dict, orient = "index")
    phi_rf.rename(index = {"A": "RF_train", "B": "RF_test"}, inplace = True)
    
    # save the model to disk
    filename = r'I:\Research\Streamlit\Titanic\models\finalized_model_rf.sav'
    pickle.dump(search.best_estimator_, open(filename, 'wb'))
 
    # load the model from disk
    loaded_model_rf = pickle.load(open(filename, 'rb'))

    
    return table_cv_rf, table_test_rf, phi_rf.transpose(),y_rf_main, loaded_model_rf

table_cv_rf, table_test_rf, phi_rf,y_rf_main, loaded_model_rf = RandomForest(df_train,df_test)

#pre = loaded_model_rf.predict(np.array([0,1,10,2,0,0,1,0,0,0,1,0,0,1]).reshape(1,-1))
#print(pre)
















