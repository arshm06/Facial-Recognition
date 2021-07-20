#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

faces = pd.read_csv('FGNet-LOPO.csv')


# In[2]:


faces['ageclass'] = faces.age.apply(
    lambda r: 0 if r < 18 else 1
).astype(int)
faces.head()


# In[3]:



x = faces.drop(["age", "ID", "Gender_0M_1F"], axis = 1)
y = faces[["age", "ID", "Gender_0M_1F", "ageclass"]]
y_young = y[y.ageclass == 0]
y_old = y[y.ageclass == 1]
x_young = x[x.ageclass == 0].drop(["ageclass"], axis = 1)
x_old = x[x.ageclass == 1].drop(["ageclass"], axis = 1)
x = x.drop(["ageclass"], axis = 1)
x_young_train, x_young_test, y_young_train, y_young_test = train_test_split(x_young, y_young, test_size = .2)
x_old_train, x_old_test, y_old_train, y_old_test = train_test_split(x_old, y_old, test_size = .2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)


# In[4]:


lis = []

for x in range(10):
    x = faces.drop(["age", "ID", "Gender_0M_1F"], axis = 1)
    y = faces[["age", "ID", "Gender_0M_1F", "ageclass"]]
    y_young = y[y.ageclass == 0]
    y_old = y[y.ageclass == 1]
    x_young = x[x.ageclass == 0].drop(["ageclass"], axis = 1)
    x_old = x[x.ageclass == 1].drop(["ageclass"], axis = 1)
    x = x.drop(["ageclass"], axis = 1)
    x_young_train, x_young_test, y_young_train, y_young_test = train_test_split(x_young, y_young, test_size = .2)
    x_old_train, x_old_test, y_old_train, y_old_test = train_test_split(x_old, y_old, test_size = .2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)
    logReg = LogisticRegression(solver= 'liblinear').fit(x_train, y_train["ageclass"])
    boost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=2).fit(x_train, y_train["ageclass"])
    svm = SVC(C=1000, gamma = .01, kernel='rbf', random_state=2).fit(x_train, y_train["ageclass"])
    forest = RandomForestClassifier(max_depth=2, random_state=2).fit(x_train, y_train["ageclass"])

    reg = Ridge(alpha=.1).fit(x_young_train, y_young_train["age"])
    regTwo = Ridge(alpha=.05).fit(x_old_train, y_old_train["age"])
    mae = {}
    errors = []

    for model in [svm]:
        score = 0
        total = 0
        error = 0
        for index, row in x_test.iterrows():
            row = row.values.reshape(1,-1)
            if model.predict(row) == 0:
                age = np.round(reg.predict(row))
                if age == y_test["age"][index]:
                    score += 1
                else:
                    error += abs(y_test["age"][index] - age)
                total += 1
            else:
                age = np.round(regTwo.predict(row))
                if age == y_test["age"][index]:
                    score += 1
                else:
                    error += abs(y_test["age"][index] - age)
                total += 1

    print(error/total)
    lis.append(float(error/total))
    
lis.append(np.mean(lis))


# In[5]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

linReg = LinearRegression().fit(x_young_train, y_young_train["age"])
KNN = KNeighborsRegressor(n_neighbors=2).fit(x_young_train, y_young_train["age"])
tree = DecisionTreeRegressor(max_depth=5).fit(x_young_train, y_young_train["age"])
bag = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=2).fit(x_young_train, y_young_train["age"])
randFor = RandomForestRegressor(max_depth=2, random_state=2).fit(x_young_train, y_young_train["age"])
boost = GradientBoostingRegressor(random_state=2).fit(x_young_train, y_young_train["age"])
ridge = Ridge(alpha=.5).fit(x_young_train, y_young_train["age"])
svm = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)).fit(x_young_train, y_young_train["age"])

errors = []
maeOne = {}
total = 0
error = 0
    
for model in [linReg, KNN, tree, bag, randFor, boost, ridge, svm]:
    total = 0
    error = 0
    for index, row in x_young_test.iterrows():
        row = row.values.reshape(1,-1)
        total += 1
        error += abs(y_young_test["age"][index] - np.round(model.predict(row)))
    errors.append(error/total)
    
modelNames = ["Linear Regression", "KNN", "Decision Tree", "Bagging", "Random Forrest", "Boosting", "Ridge", "SVM"]
for index in range(8):
    maeOne[modelNames[index]] = float(errors[index])
    
import seaborn as sns
plt.subplots(figsize = (12,12))
ax = sns.barplot(x = ["Linear Regression", "KNN", "Decision Tree", "Bagging", "Random Forrest", "Boosting", "Ridge", "SVM"], y = list(maeOne.values()), palette="Blues_d") 
plt.xlabel('Algorithm for Regression 1')
plt.ylabel('Mean Absolute Error')
for index, value in enumerate(list(maeOne.values())):
    plt.text(index-.15, value+.02,
             str(round(value, ndigits = 4)))
    
plt.savefig('SVSM Regressor 1 Hybrid')


# In[6]:


from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 
grid = GridSearchCV(Ridge(random_state=2), param_grid, refit = True, verbose = 3)
grid.fit(x_young_train, y_young_train["age"])

print(grid.best_params_)
print(grid.best_estimator_)


# In[7]:


boost = GradientBoostingRegressor(random_state=2).fit(x_young_train, y_young_train["age"])


# In[8]:


bag = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=2).fit(x_young_train, y_young_train["age"])


# In[9]:


linReg = LinearRegression().fit(x_young_train, y_young_train["age"])


# In[10]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

linReg = LinearRegression().fit(x_old_train, y_old_train["age"])
KNN = KNeighborsRegressor(n_neighbors=2).fit(x_old_train, y_old_train["age"])
tree = DecisionTreeRegressor(max_depth=5).fit(x_old_train, y_old_train["age"])
bag = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=2).fit(x_old_train, y_old_train["age"])
randFor = RandomForestRegressor(max_depth=2, random_state=2).fit(x_old_train, y_old_train["age"])
boost = GradientBoostingRegressor(random_state=2).fit(x_old_train, y_old_train["age"])
ridge = Ridge(alpha=1.0).fit(x_old_train, y_old_train["age"])
svm = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)).fit(x_old_train, y_old_train["age"])

errors = []
maeOne = {}
total = 0
error = 0
    
for model in [linReg, KNN, tree, bag, randFor, boost, ridge, svm]:
    total = 0
    error = 0
    for index, row in x_old_test.iterrows():
        row = row.values.reshape(1,-1)
        total += 1
        error += abs(y_old_test["age"][index] - np.round(model.predict(row)))
    errors.append(error/total)
    
modelNames = ["Linear Regression", "KNN", "Decision Tree", "Bagging", "Random Forrest", "Boosting", "Ridge", "SVM"]
for index in range(8):
    maeOne[modelNames[index]] = float(errors[index])
    
import seaborn as sns
plt.subplots(figsize = (12,12))
ax = sns.barplot(x = ["Linear Regression", "KNN", "Decision Tree", "Bagging", "Random Forrest", "Boosting", "Ridge", "SVM"], y = list(maeOne.values()), palette="Blues_d") 
plt.xlabel('Algorithm for Regression 2')
plt.ylabel('Mean Absolute Error')
for index, value in enumerate(list(maeOne.values())):
    plt.text(index-.2, value+.06,
             str(round(value, ndigits = 4)))
    
plt.savefig('SVSM Regressor 2 Hybrid')


# In[11]:


import seaborn as sns
print(lis)
plt.subplots(figsize = (12,14))
ax = sns.barplot(x = ["Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5","Trial 6", "Trial 7", "Trial 8", "Trial 9", "Trial 10", "Accuracy"], y = list(lis), palette="Blues_d") 
plt.xlabel('Trials for Hierarchical Model')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)

for index, value in enumerate(list(lis)):
    plt.text(index-.2, value+.04,
             str(round(value, ndigits = 4)))


# In[ ]:




