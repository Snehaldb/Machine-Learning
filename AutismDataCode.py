#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


# In[47]:


def normalize(X):
    Xmin = np.amin(X)
    Xmax = np.amax(X)
    X = (X-Xmin)/(Xmax-Xmin)
    return X


# Add missing data to ethnicity , relation features

# In[48]:


#read csv
autism_data = pd.read_csv("Data_Autism.csv")


# In[49]:


autism_data


# In[50]:


columns_dont_want = ["ethnicity", "relation","Class/ASD {NO,YES}"]
select = [x for x in autism_data.columns if x not in columns_dont_want]

y_ethnicity = autism_data.loc[:, 'ethnicity']
X_ethnicity = autism_data.loc[:, select]
X_numbers = X_ethnicity.select_dtypes(include=[np.number])

#Give labels and use df.apply() to apply le.fit_transform to all columns
le = preprocessing.LabelEncoder()
X_dataset_ohe = X_ethnicity.select_dtypes(include=[object]).apply(le.fit_transform).values

#Apply one hot encoding to create vector
enc = preprocessing.OneHotEncoder()
enc.fit(X_dataset_ohe)
X_onehotlabels = enc.transform(X_dataset_ohe).toarray()

y_ethnicity = y_ethnicity.values
X_ethnicity = X_onehotlabels
X_ethnicity = np.hstack((X_ethnicity,X_numbers))


# In[51]:


x_ethnicityTest = []
y_ethnicityTest = []
x_ethnicityData = []
y_ethnicityData = []
for xi,yi in zip(X_ethnicity,y_ethnicity):
    if(yi=='?'):
        x_ethnicityTest.append(xi)
        y_ethnicityTest.append(yi)
    else:
        x_ethnicityData.append(xi)
        y_ethnicityData.append(yi)
y_ethnicityData = np.array(y_ethnicityData)
x_ethnicityData = np.array(x_ethnicityData)
x_ethnicityTest = np.array(x_ethnicityTest)
y_ethnicityTest = np.array(y_ethnicityTest)


# Running Logistic Regression to fill ETHNICITY MISSING VALUES

# In[52]:


from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(solver='saga',multi_class='ovr',max_iter=1000,penalty='l1')


# In[8]:


logReg.fit(x_ethnicityData,y_ethnicityData)
valYPredicted = logReg.predict(x_ethnicityData)
valYTrainPredicted = logReg.predict(x_ethnicityData)
print('Accuracy of Train data: %.2f' %accuracy_score(y_ethnicityData, valYTrainPredicted))
#print('Accuracy of Validation data: %.2f' %accuracy_score(y_Val, valYPredicted))


# Running Random Forest Classifier to fill ETHNICITY MISSING VALUES 

# In[53]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 5, min_samples_split=10, max_depth =15)
rfc.fit(x_ethnicityData,y_ethnicityData)
valYPredicted = rfc.predict(x_ethnicityData)
print('Accuracy of Training data: %.2f' %accuracy_score(y_ethnicityData, valYPredicted))


# Better Accuracy with Random Forest. Using it to get the missing values.

# In[54]:


valuesofEth = rfc.predict(x_ethnicityTest)


# In[55]:


i=0
for index, row in autism_data.iterrows():
    if(row['ethnicity']=='?' and i<valuesofEth.shape[0]):
        autism_data['ethnicity'][index]=valuesofEth[i]
        i=i+1


# In[56]:


autism_data


# Normalizing Age Numeric and result numeric fields

# In[57]:


ageData = np.array(autism_data.loc[:, 'age numeric'])
ageData = normalize(ageData)


# Updating dataframe

# In[58]:


newdf = pd.DataFrame({'age numeric': ageData})
autism_data.update(newdf)
autism_data


# In[59]:


autism_data = autism_data.drop(['relation','used_app_before {no,yes}','result numeric'],axis = 1 )
autism_data


# In[61]:


#create X and Y dataset
y_dataset = autism_data.iloc[:, 16]
X_dataset = autism_data.iloc[:, 0:16].select_dtypes(include=[object])
X_numbers = autism_data.select_dtypes(include=[np.number])


#Give labels and use df.apply() to apply le.fit_transform to all columns
le = preprocessing.LabelEncoder()
X_dataset_ohe = X_dataset.apply(le.fit_transform)


# In[64]:


X_dataset_ohe


# Performing Feature Selection Using RandomForestClassifier

# In[62]:


X_featureImp = np.hstack((X_dataset_ohe,X_numbers))
X_train, X_test, y_train, y_test = train_test_split(X_featureImp, y_dataset, test_size=0.30,random_state=1)
rfc = RandomForestClassifier(n_estimators = 20,random_state=1,max_depth=1)
rfc.fit(X_train,y_train)
rfc.feature_importances_


# The feature selection shows that the following columns are more important -  A2_Score {0,1}	A3_Score {0,1}	A4_Score {0,1}	A5_Score {0,1}	A6_Score {0,1}	A9_Score {0,1}	A10_Score {0,1}

# Training the Filled dataset

# In[90]:


#Apply one hot encoding to create vector
enc = preprocessing.OneHotEncoder()
enc.fit(X_dataset_ohe)
X_onehotlabels = enc.transform(X_dataset_ohe).toarray()
X= np.hstack((X_onehotlabels,X_numbers))

#create the dataframe
x_dataframe = pd.DataFrame(X)
y_dataframe = pd.DataFrame(y_dataset)


# create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y_dataset, test_size=0.30,random_state=1)


# Analyzing the Train and Test data

# In[99]:


unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))


# In[100]:


unique, counts = np.unique(y_test, return_counts=True)
dict(zip(unique, counts))


# Using Stratified Splitting

# In[104]:


sss = StratifiedShuffleSplit(n_splits=2, test_size=0.30, random_state=1)
for train_index, test_index in sss.split(X, y_dataset):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_dataset[train_index], y_dataset[test_index]
    
unique, counts = np.unique(y_tests, return_counts=True)
dict(zip(unique, counts))


# In[105]:


d_tree = DecisionTreeClassifier()
d_tree.fit(X_train, y_train)
y_pred_test = d_tree.predict(X_test)
y_pred_train = d_tree.predict(X_train)

print('Accuracy of Test data Using Decision Tree: %.2f' %accuracy_score(y_test, y_pred_test))


# In[119]:


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
# Show confusion matrix
plt.matshow(cm)
plt.title('Decision Tree Confusion matrix',y=1.20)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Running Logistic Regression on the dataset

# In[118]:


regr_2 = LogisticRegression(solver="liblinear", multi_class="ovr")
regr_2.fit(X_train, y_train)
y_pred_test_lr = regr_2.predict(X_test)
y_pred_train_lr = regr_2.predict(X_train)

print('Accuracy of Training data using Logistic Regression: %.2f' %accuracy_score(y_train, y_pred_train_lr))
print('Accuracy of Test data using Logistic Regression: %.2f' %accuracy_score(y_test, y_pred_test_lr))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_test_lr)
print(cm)
# Show confusion matrix
plt.matshow(cm)
plt.title('Logistic Regression Confusion matrix',y=1.20)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Running MLP Classifier on the dataset without K-Folds

# In[113]:


mlp = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[10,10], max_iter=400, activation='logistic')
mlp.fit(X_train, y_train)

print("Accuracy of Training set using MLP Classifier: %f" % mlp.score(X_train, y_train))
print("Accuracy of Test set using MLP Classifier: %f" % mlp.score(X_test, y_test))


# Using K-Fold with MLP Classifier
# 

# In[114]:


kf = KFold(n_splits=10)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,20,20,30,30), random_state=1)
training_score = []
k_size = []

for train_indices, validation_indices in kf.split(X_train,y_train):
    clf.fit(X[train_indices], y_dataset[train_indices])
    print(clf.score(X[validation_indices], y_dataset[validation_indices]))
    


# In[115]:


y_pred_test = clf.predict(X_test)
print('Accuracy of Test data using k-fold Cross Validation with MLPClassifier: %.2f' %accuracy_score(y_test, y_pred_test))
clf.score(X_test, y_test)


# In[117]:


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
# Show confusion matrix
plt.matshow(cm)
plt.title('K-Fold Confusion matrix',y=1.20)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

